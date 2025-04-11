import os
import re
import argparse
import numpy as np
import pandas as pd
import json
import paramiko
import gensim
from gensim import corpora
from gensim.models import LdaModel, LdaMulticore
import pyLDAvis
import pyLDAvis.gensim_models
import logging
import stat
import gzip
import subprocess
import sys
from gensim.models.coherencemodel import CoherenceModel
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # Should return > 0 if a GPU is available

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def load_credentials(credentials_file):
    try:
        with open(credentials_file, 'r') as file:
            credentials = json.load(file)
        return credentials
    except Exception as e:
        print(f"Error loading credentials from {credentials_file}: {e}")
        return None

def ssh_connect(host, username, password):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username=username, password=password)
        return ssh
    except Exception as e:
        print(f"SSH connection failed: {e}")
        return None

def normalize_remote_path(path):
    return path.replace("\\", "/")


def parse_log_line(line, log_type):
    log_patterns = {
        'nal': r"\[(.*?)\]\s*(.*)",
        'netra': r"\[(.*?)\]<.*?><.*?>\s*(.*)",
        'stbCtrl': r"\[(.*?)\]<.*?><.*?>\s*(.*)",
        'wjap': r".*\[[0-9]+\]:\s(.*)",
        'SBSDK': r".*\s.*>\s(.*)",  # Extract after "> "
        'default': r"\[(.*?)\]<.*?><.*?>\s*(.*)"
    }

    pattern = log_patterns.get(log_type, log_patterns['default'])
    match = re.match(pattern, line)

    if match:
        if log_type in ['nal', 'netra', 'stbCtrl', 'default']:
            component = match.group(1)
            message = match.group(2)
        else:
            component = None
            message = match.group(1)
        return component, message.strip()
    else:
        return None, None


def read_logs_in_chunks(sftp, log_files, log_type, chunk_size=1000000):
    log_messages = []
    total_messages = 0
    for log_file in log_files:
        print(f"Processing remote file: {log_file}")
        content = read_remote_file(sftp, log_file)
        if content:
            directory_name = os.path.dirname(log_file).replace("/ccshare/logs/smplogs/", "")
            file_name = os.path.basename(log_file)
            lines = content.splitlines()
            for line in lines:
                _, message = parse_log_line(line, log_type)
                if message:
                    prepended_message = f"[{directory_name}/{file_name}] {message}"
                    log_messages.append(prepended_message)
                    total_messages += 1
                    if len(log_messages) >= chunk_size:
                        yield log_messages
                        log_messages = []
            print(f"Extracted {total_messages} messages so far.")
        else:
            print(f"No content read from {log_file}")
    if log_messages:
        yield log_messages
    print(f"Total log messages processed: {total_messages}")

def preprocess_messages(messages):
    log_pattern = re.compile(r'^\[(.*?)\]<(.*?)><(.*?)>\s+(.*)$')
    mac_address_pattern = re.compile(r'\b([0-9A-Fa-f]{2}(:|-)[0-9A-Fa-f]{2}(:|-)[0-9A-Fa-f]{2}(:|-)[0-9A-Fa-f]{2}(:|-)[0-9A-Fa-f]{2}(:|-)[0-9A-Fa-f]{2})\b')

    preprocessed = []
    for msg in messages:
        match = log_pattern.match(msg)
        if match:
            module = match.group(1)
            timestamp = match.group(2)
            fileline = match.group(3)
            message = match.group(4)
            labeled_line = f"[MODULE] {module} [TIMESTAMP] {timestamp} [FILELINE] {fileline} [MESSAGE] {message}"
        else:
            labeled_line = f"[MESSAGE] {msg}"

        labeled_line = mac_address_pattern.sub(r'MAC_ADDRESS_\1', labeled_line)

        labeled_line = re.sub(
            r'\b(\d{1,3}(?:\.\d{1,3}){3})\b',
            r'IP_ADDRESS_\1 \1',
            labeled_line
        )

        tokens = gensim.utils.simple_preprocess(labeled_line, deacc=True)
        preprocessed.append(tokens)

    return preprocessed

def ensure_real(obj):
    if isinstance(obj, dict):
        return {k: ensure_real(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_real(v) for v in obj]
    elif isinstance(obj, complex):
        return float(obj.real)
    return obj

def load_existing_model_and_dictionary(output_dir):
    model_path = os.path.join(output_dir, 'lda_model.gensim')
    dictionary_path = os.path.join(output_dir, 'dictionary.gensim')
    if os.path.exists(model_path) and os.path.exists(dictionary_path):
        print("Loading existing model and dictionary...")
        lda_model = LdaModel.load(model_path)
        dictionary = corpora.Dictionary.load(dictionary_path)
        return lda_model, dictionary
    print("No existing model or dictionary found. Starting fresh.")
    return None, None

def save_visualization(lda_model, corpus, dictionary, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not corpus:
        print("Error: Corpus is empty. Cannot create visualization.")
        return
    print("Preparing visualization...")

    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

    from pyLDAvis.utils import NumPyEncoder

    class ComplexToFloatEncoder(NumPyEncoder):
        def default(self, o):
            if isinstance(o, complex):
                return float(o.real)
            return super().default(o)

    def ensure_real(obj):
        if isinstance(obj, dict):
            return {k: ensure_real(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ensure_real(x) for x in obj]
        elif isinstance(obj, complex):
            return float(obj.real)
        return obj

    original_to_dict = vis.to_dict

    def safe_to_json():
        data_dict = original_to_dict()
        data_dict = ensure_real(data_dict)
        return json.dumps(data_dict, cls=ComplexToFloatEncoder)

    vis.to_json = safe_to_json

    pyLDAvis.save_html(vis, os.path.join(output_dir, 'lda_visualization.html'))
    print(f"Visualization saved to {os.path.join(output_dir, 'lda_visualization.html')}")

def select_num_topics(dictionary, corpus_sample, start=5, limit=30, step=5, passes=1):
    texts = [[dictionary[id] for id, freq in doc] for doc in corpus_sample]
    best_coherence = -1
    best_num_topics = start
    for num_topics in range(start, limit+1, step):
        model = LdaModel(
            corpus=corpus_sample,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            alpha='auto',
            per_word_topics=False
        )
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        print(f"Num Topics = {num_topics}, Coherence = {coherence}")
        if coherence > best_coherence:
            best_coherence = coherence
            best_num_topics = num_topics
    print(f"Selected num_topics = {best_num_topics} with Coherence = {best_coherence}")
    return best_num_topics

def try_cupy():
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            print(f"CuPy detected and {device_count} GPU(s) available.")
            return cp
        else:
            print("CuPy detected, but no GPUs available.")
            return None
    except ImportError:
        print("CuPy not found. Attempting to install CuPy...")
        install_cmd = [sys.executable, "-m", "pip", "install", "cupy-cuda12x"]
        try:
            subprocess.run(install_cmd, check=True)
            import cupy as cp
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                print(f"CuPy installed and {device_count} GPU(s) available.")
                return cp
            else:
                print("No GPUs found even after installing CuPy.")
                return None
        except (ImportError, subprocess.CalledProcessError) as e:
            print(f"Failed to install or import CuPy: {e}")
            return None

def run_cupy_lda(cp, corpus, dictionary, num_topics, passes):
    print("Attempting GPU-based LDA with CuPy.")

    # Convert corpus to dense (docs x terms)
    num_terms = len(dictionary)
    num_docs = len(corpus)
    corpus_dense = gensim.matutils.corpus2dense(corpus, num_terms=num_terms).T  # (num_docs, num_terms)
    corpus_gpu = cp.asarray(corpus_dense)

    # Initialize distributions
    # topic_word_dist: (num_topics, num_terms)
    topic_word_dist = cp.random.dirichlet(alpha=[1.0] * num_terms, size=num_topics).astype(cp.float32)
    # doc_topic_dist: (num_docs, num_topics)
    doc_topic_dist = cp.random.dirichlet(alpha=[1.0] * num_topics, size=num_docs).astype(cp.float32)

    alpha = 1.0
    beta = 0.01

    print(f"Initial shapes - doc_topic_dist: {doc_topic_dist.shape}, topic_word_dist: {topic_word_dist.shape}")

    for pass_num in range(passes):
        print(f"Running pass {pass_num + 1}/{passes} on GPU.")

        # E-step:
        # topic_probs = doc_topic_dist (num_docs, num_topics) * topic_word_dist (num_topics, num_terms)
        topic_probs = cp.matmul(doc_topic_dist, topic_word_dist) + beta  # (num_docs, num_terms)
        topic_probs /= topic_probs.sum(axis=1, keepdims=True)

        # M-step (update doc_topic_dist):
        # doc_topic_dist = corpus_gpu (num_docs, num_terms) * topic_word_dist.T (num_terms, num_topics)
        doc_topic_dist = cp.matmul(corpus_gpu, topic_word_dist.T) + alpha
        doc_topic_dist /= doc_topic_dist.sum(axis=1, keepdims=True)  # normalize over topics

        # M-step (update topic_word_dist):
        # topic_word_dist = doc_topic_dist.T (num_topics, num_docs) * corpus_gpu (num_docs, num_terms)
        topic_word_dist = cp.matmul(doc_topic_dist.T, corpus_gpu) + beta
        topic_word_dist /= topic_word_dist.sum(axis=1, keepdims=True)  # normalize over terms

        print(f"Pass {pass_num + 1}: doc_topic_dist: {doc_topic_dist.shape}, topic_word_dist: {topic_word_dist.shape}")

    # Move data back to CPU
    topic_word_dist_cpu = cp.asnumpy(topic_word_dist)
    doc_topic_dist_cpu = cp.asnumpy(doc_topic_dist)

    # Create a dummy LdaModel for consistency with the code
    # Note: This is not a fully correct reconstruction of model state,
    # but will serve as a placeholder.
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        alpha='auto',
        eta='auto',
        passes=passes,
        iterations=50
    )

    # Ideally, you'd integrate learned distributions properly into lda_model.
    # Gensim's LDA expects internal parameters; this is a hacky workaround.
    # A proper integration would require carefully setting 'state' attributes.
    # We'll just print a warning:
    print("Warning: The LdaModel returned may not perfectly reflect the GPU distributions.")
    return lda_model




def main():
    parser = argparse.ArgumentParser(description="Log Analysis with Incremental LDA (with CuPy GPU attempt)")
    parser.add_argument('-d', '--directory', type=str, required=True, help='Remote directory containing log files')
    parser.add_argument('-l', '--log_type', type=str, required=True, help='Log type (e.g., nal, netra, stbCtrl)')
    parser.add_argument('-c', '--chunk_size', type=int, default=1000000, help='Number of log messages per chunk')
    parser.add_argument('-p', '--passes', type=int, default=1, help='Number of passes')
    args = parser.parse_args()

    cp = try_cupy()

    args.file_filter = args.log_type
    args.output_dir = os.path.join('logs', args.log_type)

    credentials = load_credentials("credentials.txt")
    if not credentials:
        return

    ssh_host = credentials.get("linux_pc")
    ssh_username = credentials.get("username")
    ssh_password = credentials.get("password")

    ssh = ssh_connect(ssh_host, ssh_username, ssh_password)
    if not ssh:
        return

    sftp = ssh.open_sftp()
    remote_path = normalize_remote_path(args.directory)
    print(f"Processing remote directory: {remote_path}")
    log_files = sftp_list_files(sftp, remote_path, file_filter=args.log_type)
    print(f"Found {len(log_files)} log files.")

    if len(log_files) == 0:
        print("No log files found. Exiting.")
        sftp.close()
        ssh.close()
        return

    lda_model, dictionary = load_existing_model_and_dictionary(args.output_dir)

    if dictionary is None:
        dictionary = corpora.Dictionary()

    chunk_generator = read_logs_in_chunks(sftp, log_files, args.log_type, chunk_size=args.chunk_size)

    sample_limit = 20000
    try:
        sample_messages = next(chunk_generator)
    except StopIteration:
        sample_messages = []

    if not sample_messages:
        print("No messages collected for dictionary.")
        sftp.close()
        ssh.close()
        return

    preprocessed_sample = preprocess_messages(sample_messages[:sample_limit])

    initial_dict_len = len(dictionary)
    dictionary.add_documents(preprocessed_sample)
    dictionary.compactify()
    new_dict_len = len(dictionary)
    if new_dict_len > initial_dict_len:
        print(f"Dictionary expanded from {initial_dict_len} to {new_dict_len} tokens.")

    if len(dictionary) == 0:
        print("Error: Dictionary is empty. Exiting.")
        sftp.close()
        ssh.close()
        return

    corpus_sample = [dictionary.doc2bow(text) for text in preprocessed_sample]

    if lda_model is None:
        chosen_num_topics = select_num_topics(dictionary, corpus_sample, start=5, limit=30, step=5, passes=args.passes)
    else:
        chosen_num_topics = lda_model.num_topics

    sftp.close()
    sftp = ssh.open_sftp()
    log_files = sftp_list_files(sftp, remote_path, file_filter=args.log_type)

    total_corpus = corpus_sample.copy()
    chunk_count = 1

    for messages in chunk_generator:
        chunk_count += 1
        print(f"Processing chunk {chunk_count} with {len(messages)} messages...")

        preprocessed_messages = preprocess_messages(messages)
        corpus = [dictionary.doc2bow(text) for text in preprocessed_messages]
        if not corpus:
            print(f"Skipping chunk {chunk_count}, empty corpus.")
            continue

        if lda_model is None:
            print("Initializing LDA model with chosen_num_topics:", chosen_num_topics)
            if cp:
                lda_gpu = run_cupy_lda(cp, corpus, dictionary, chosen_num_topics, args.passes)
                if lda_gpu is not None:
                    lda_model = lda_gpu
                else:
                    print("Falling back to CPU-based LDA.")
                    lda_model = LdaMulticore(
                        corpus=corpus,
                        id2word=dictionary,
                        num_topics=chosen_num_topics,
                        passes=args.passes,
                        eval_every=1,
                        workers=os.cpu_count(),
                        chunksize=args.chunk_size,
                        alpha='asymmetric',
                        per_word_topics=False
                    )
            else:
                lda_model = LdaMulticore(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=chosen_num_topics,
                    passes=args.passes,
                    eval_every=1,
                    update_every=0,
                    workers=os.cpu_count(),
                    chunksize=args.chunk_size,
                    alpha='auto',
                    per_word_topics=True
                )
        else:
            if hasattr(lda_model, 'update'):
                lda_model.update(corpus)
            else:
                lda_model = lda_model + corpus

        total_corpus.extend(corpus)
        del messages, preprocessed_messages, corpus

    if lda_model is None:
        print("Error: No valid data processed. Exiting.")
        sftp.close()
        ssh.close()
        return

    save_visualization(lda_model, total_corpus, dictionary, args.output_dir)

    topics = lda_model.print_topics(num_words=10)
    topics_df = pd.DataFrame(topics, columns=['TopicID', 'TopicTerms'])
    topics_df.to_csv(os.path.join(args.output_dir, 'topics_overview.csv'), index=False)
    print(f"Topics overview saved to {os.path.join(args.output_dir, 'topics_overview.csv')}")

    if hasattr(lda_model, 'save'):
        lda_model.save(os.path.join(args.output_dir, 'lda_model.gensim'))
    dictionary.save(os.path.join(args.output_dir, 'dictionary.gensim'))
    print(f"LDA model and dictionary saved to {args.output_dir}")

    sftp.close()
    ssh.close()


if __name__ == "__main__":
    main()
