import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from notion_client import Client

def create_rich_block_structure(paper):
    """Creates a comprehensive block structure for the Resource page."""
    blocks = [
        {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": f"📄 Title: {paper['title']}\n👤 Authors: {paper['authors']} ({paper['year']})\n🏛️ Venue: {paper['venue']}\n🔗 Direct Link: "
                        }
                    },
                    {
                        "type": "text",
                        "text": {
                            "content": paper["url"],
                            "link": {"url": paper["url"]}
                        }
                    }
                ],
                "icon": {"emoji": "📄"}
            }
        },
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": "🎯 Executive Summary"}}]
            }
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": paper["summary"]}}]
            }
        },
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": "💡 Key Concepts & Core Contributions"}}]
            }
        }
    ]

    for pt in paper["key_points"]:
        blocks.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": pt}}]
            }
        })

    # Bookmark / Link Block
    blocks.append({
        "object": "block",
        "type": "bookmark",
        "bookmark": {
            "url": paper["url"]
        }
    })

    # Deep Reading Checklist
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "📝 Deep Reading & Analysis Checklist"}}]
        }
    })
    blocks.append({
        "object": "block",
        "type": "to_do",
        "to_do": {
            "rich_text": [{"type": "text", "text": {"content": "Read abstract, introduction, and problem formulation."}}],
            "checked": False
        }
    })
    blocks.append({
        "object": "block",
        "type": "to_do",
        "to_do": {
            "rich_text": [{"type": "text", "text": {"content": "Trace and derive core mathematical formulations or architectural diagrams."}}],
            "checked": False
        }
    })
    blocks.append({
        "object": "block",
        "type": "to_do",
        "to_do": {
            "rich_text": [{"type": "text", "text": {"content": "Document key insights, historical significance, and modern applications in notes."}}],
            "checked": False
        }
    })

    return blocks


def main():
    notion_token = os.getenv("NOTION_TOKEN")
    if not notion_token:
        print("Error: NOTION_TOKEN not set in environment.")
        sys.exit(1)

    client = Client(auth=notion_token)

    resources_db_id = "54d38af8-cb58-82d6-8487-81ed4c16cd0f"
    tasks_db_id = "3b938af8-cb58-803a-b959-f1a85a4bceb3"
    subjects_db_id = "90538af8-cb58-8225-bb41-812b0ab3bf25"

    print("--- Notion Database Connections ---")
    print(f"Resources DB ID: {resources_db_id}")
    print(f"Tasks DB ID:     {tasks_db_id}")
    print(f"Subjects DB ID:  {subjects_db_id}")

    SUBJECT_MAP = {
        "foundations": "3c138af8-cb58-81f3-adb0-c6b287089e8b",   # Foundations of Computing & Information Theory
        "neural_nets": "3c138af8-cb58-8182-a6d1-d99e14626ecd",   # Neural Networks & Connectionism
        "distributed": "3c138af8-cb58-8173-a7c2-e17115ac4f94",   # Distributed Systems & Web-Scale Architecture
        "deep_learning": "3c138af8-cb58-812f-993d-d83f18d9612c",  # Deep Learning, Transformers & Foundation Models
    }

    papers = [
        {
            "title": "On Computable Numbers, with an Application to the Entscheidungsproblem",
            "short_title": "On Computable Numbers (Alan Turing, 1936)",
            "authors": "Alan Turing",
            "year": 1936,
            "venue": "Proceedings of the London Mathematical Society",
            "url": "https://www.cs.virginia.edu/~robins/Turing_Paper_1936.pdf",
            "subject_key": "foundations",
            "task_title": "Read & Annotate: On Computable Numbers (Alan Turing, 1936)",
            "task_desc": "Universal Turing Machine (UTM), halting problem undecidability, and computable real numbers.",
            "priority": "High",
            "effort": "Large",
            "summary": "Introduces the abstract computing machine (Universal Turing Machine) and proves that Hilbert's Entscheidungsproblem has no general algorithmic decision procedure.",
            "key_points": [
                "Definition of 'computable numbers' as real numbers whose decimals are calculable by finite mechanical means.",
                "Specification of the Universal Turing Machine (UTM) as the foundational architecture for general-purpose computers.",
                "Proof that the Halting Problem is undecidable using Cantor's diagonal argument.",
                "Established the Church-Turing thesis defining the mathematical boundaries of computation."
            ]
        },
        {
            "title": "A Mathematical Theory of Communication",
            "short_title": "A Mathematical Theory of Communication (Claude Shannon, 1948)",
            "authors": "Claude Shannon",
            "year": 1948,
            "venue": "Bell System Technical Journal",
            "url": "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf",
            "subject_key": "foundations",
            "task_title": "Read & Annotate: A Mathematical Theory of Communication (Claude Shannon, 1948)",
            "task_desc": "Information entropy, the bit as fundamental unit, channel capacity, and source/channel coding theorems.",
            "priority": "High",
            "effort": "Large",
            "summary": "Founded the field of Information Theory, mathematically formalizing communication, quantifying information via entropy, and establishing fundamental limits on data compression and channel capacity.",
            "key_points": [
                "Introduced the 'bit' as the fundamental quantitative unit of information.",
                "Formulated Information Entropy H = -sum(p_i * log2(p_i)) to measure uncertainty and information content.",
                "Proved the Source Coding Theorem defining the theoretical lower bound for lossless compression.",
                "Proved the Noisy-Channel Coding Theorem establishing maximum error-free transmission rate (Channel Capacity C = B * log2(1 + S/N))."
            ]
        },
        {
            "title": "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain",
            "short_title": "The Perceptron (Frank Rosenblatt, 1958)",
            "authors": "Frank Rosenblatt",
            "year": 1958,
            "venue": "Psychological Review",
            "url": "https://cns-classes.bu.edu/cn510-lectures/rosenblatt-1958.pdf",
            "subject_key": "neural_nets",
            "task_title": "Read & Annotate: The Perceptron (Frank Rosenblatt, 1958)",
            "task_desc": "Linear threshold units, perceptron convergence theorem, biological connectionism.",
            "priority": "Medium",
            "effort": "Medium",
            "summary": "Introduced the Perceptron, the earliest foundational computational model for artificial neural networks and supervised pattern recognition.",
            "key_points": [
                "Linear threshold unit architecture combining sensory (S), association (A), and response (R) units.",
                "Perceptron learning rule iteratively adjusting weights based on classification errors.",
                "Perceptron Convergence Theorem: guarantees learning a separating hyperplane if data is linearly separable.",
                "Pioneered connectionist theory linking biological neural modeling to machine learning."
            ]
        },
        {
            "title": "Perceptrons: An Introduction to Computational Geometry",
            "short_title": "Perceptrons (Marvin Minsky & Seymour Papert, 1969)",
            "authors": "Marvin Minsky and Seymour Papert",
            "year": 1969,
            "venue": "MIT Press",
            "url": "https://web.media.mit.edu/~minsky/papers/Perceptrons.html",
            "subject_key": "neural_nets",
            "task_title": "Read & Annotate: Perceptrons (Minsky & Papert, 1969)",
            "task_desc": "Geometric limits of single-layer perceptrons, XOR limitation, topological predicates.",
            "priority": "Medium",
            "effort": "Medium",
            "summary": "Rigorous mathematical analysis of the computational limitations of single-layer perceptrons, proving their inability to compute non-linearly separable functions like XOR.",
            "key_points": [
                "Proved that single-layer perceptrons cannot compute the XOR (exclusive-or) logic function.",
                "Demonstrated limits on topological predicates (e.g. determining connectivity or parity in figures).",
                "Emphasized the theoretical requirement for multi-layer networks while noting the lack of training algorithms at the time.",
                "Historically shifted AI research toward symbolic methods, ushering in the first AI Winter."
            ]
        },
        {
            "title": "Time, Clocks, and the Ordering of Events in a Distributed System",
            "short_title": "Time, Clocks, and Ordering of Events (Leslie Lamport, 1978)",
            "authors": "Leslie Lamport",
            "year": 1978,
            "venue": "Communications of the ACM (CACM)",
            "url": "https://lamport.azurewebsites.net/pubs/time-clocks.pdf",
            "subject_key": "distributed",
            "task_title": "Read & Annotate: Time, Clocks, and Ordering of Events (Leslie Lamport, 1978)",
            "task_desc": "Happened-before relation (->), logical clocks, total ordering, distributed synchronization.",
            "priority": "High",
            "effort": "Medium",
            "summary": "Foundational paper in distributed computing introducing logical timestamps to define a partial and total ordering of events without relying on physical clock synchronization.",
            "key_points": [
                "Defined the 'happened-before' relation (a -> b) formalizing causal order in distributed asynchronous systems.",
                "Introduced Lamport Logical Clocks / Timestamps to track causality between communicating processes.",
                "Extended partial causal ordering to a consistent total ordering to resolve race conditions.",
                "Constructed a distributed state machine protocol for mutual exclusion and fault-tolerant synchronization."
            ]
        },
        {
            "title": "Learning representations by back-propagating errors",
            "short_title": "Learning representations by back-propagating errors (Rumelhart et al., 1986)",
            "authors": "David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams",
            "year": 1986,
            "venue": "Nature",
            "url": "https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf",
            "subject_key": "neural_nets",
            "task_title": "Read & Annotate: Learning representations by back-propagating errors (Rumelhart et al., 1986)",
            "task_desc": "Generalized delta rule, multi-layer error gradient propagation, internal hidden representations.",
            "priority": "High",
            "effort": "Medium",
            "summary": "Demonstrated that the backpropagation algorithm efficiently trains multi-layer feedforward neural networks, enabling hidden units to learn complex non-linear representations and overcoming Minsky & Papert's critique.",
            "key_points": [
                "Formulated gradient descent via the calculus chain rule for multi-layer neural networks.",
                "Showed hidden layers automatically discover internal semantic features (e.g. family trees, symmetry).",
                "Overcame the XOR barrier and proved multi-layer networks can approximate complex functions.",
                "Established the computational cornerstone for modern deep learning optimization."
            ]
        },
        {
            "title": "The Anatomy of a Large-Scale Hypertextual Web Search Engine",
            "short_title": "The Anatomy of a Search Engine (Sergey Brin & Larry Page, 1998)",
            "authors": "Sergey Brin and Larry Page",
            "year": 1998,
            "venue": "Computer Networks and ISDN Systems (Stanford)",
            "url": "http://ilpubs.stanford.edu:8090/361/1/1998-8.pdf",
            "subject_key": "distributed",
            "task_title": "Read & Annotate: The Anatomy of a Search Engine (Brin & Page, 1998)",
            "task_desc": "PageRank random surfer model, web crawler design, forward and inverted indexing pipeline.",
            "priority": "High",
            "effort": "Medium",
            "summary": "Introduced the prototype architecture of Google, detailing the PageRank algorithm and scalable distributed systems for web crawling, indexing, and query evaluation.",
            "key_points": [
                "PageRank algorithm: modeling web graph hyperlink structure as a random walk stationary probability distribution.",
                "Anchor text association and formatting/proximity weights for high-precision search ranking.",
                "Scalable system architecture: Crawler, StoreServer, Indexer, Sorter, Barrels, and Searcher.",
                "Practical distributed engineering techniques for memory caching, disk I/O, and index compression."
            ]
        },
        {
            "title": "ImageNet Classification with Deep Convolutional Neural Networks",
            "short_title": "AlexNet / ImageNet Classification (Krizhevsky, Sutskever, Hinton, 2012)",
            "authors": "Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton",
            "year": 2012,
            "venue": "Advances in Neural Information Processing Systems (NeurIPS 2012)",
            "url": "https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf",
            "subject_key": "deep_learning",
            "task_title": "Read & Annotate: AlexNet / ImageNet Classification (Krizhevsky et al., 2012)",
            "task_desc": "GPU parallelism (CUDA), ReLU non-saturating activation, Dropout regularization, ImageNet breakthrough.",
            "priority": "High",
            "effort": "Medium",
            "summary": "Trained a massive 60-million parameter deep convolutional neural network (AlexNet) on GPUs to decisively win the ImageNet competition, sparking the modern deep learning revolution.",
            "key_points": [
                "Architecture: 8 learned layers (5 convolutional, 3 fully-connected) with 60M parameters and 650k neurons.",
                "Used non-saturating Rectified Linear Units (ReLU) achieving 6x faster convergence over tanh.",
                "Leveraged multi-GPU parallel computing (CUDA) and heavy Dropout (0.5) to prevent overfitting.",
                "Achieved top-5 error rate of 15.3% vs 26.2% for the second-place system, proving deep learning supremacy."
            ]
        },
        {
            "title": "Attention Is All You Need",
            "short_title": "Attention Is All You Need (Vaswani et al., 2017)",
            "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin",
            "year": 2017,
            "venue": "Advances in Neural Information Processing Systems (NeurIPS 2017)",
            "url": "https://arxiv.org/pdf/1706.03762.pdf",
            "subject_key": "deep_learning",
            "task_title": "Read & Annotate: Attention Is All You Need (Vaswani et al., 2017)",
            "task_desc": "Scaled Dot-Product Attention, Multi-Head Attention, Positional Encoding, Transformer architecture.",
            "priority": "High",
            "effort": "Large",
            "summary": "Proposed the Transformer architecture based entirely on self-attention mechanisms, eliminating recurrence and convolutions and establishing the blueprint for modern NLP and foundation models.",
            "key_points": [
                "Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V.",
                "Multi-Head Attention mechanism allowing parallel representation learning across multiple subspaces.",
                "Sinusoidal and learned positional encodings to inject sequence order without recurrent loops.",
                "Enabled massive training parallelization on GPUs/TPUs, leading directly to BERT, GPT, and modern LLMs."
            ]
        },
        {
            "title": "Language Models are Few-Shot Learners",
            "short_title": "Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)",
            "authors": "Tom B. Brown et al. (OpenAI)",
            "year": 2020,
            "venue": "Advances in Neural Information Processing Systems (NeurIPS 2020)",
            "url": "https://arxiv.org/pdf/2005.14165.pdf",
            "subject_key": "deep_learning",
            "task_title": "Read & Annotate: Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)",
            "task_desc": "In-context learning, Zero/One/Few-shot prompting, compute scaling laws, 175B parameter autoregressive model.",
            "priority": "High",
            "effort": "Large",
            "summary": "Demonstrated that scaling autoregressive language models to 175 billion parameters (GPT-3) results in emergent few-shot in-context learning capabilities across a vast array of tasks without gradient updates.",
            "key_points": [
                "Discovered that parameter and compute scale enables zero-shot, one-shot, and few-shot task adaptation.",
                "Eliminated the requirement for fine-tuning weights on task-specific training sets.",
                "Comprehensive evaluation across NLP benchmarks, arithmetic, translation, and code generation.",
                "Formally established prompt engineering and in-context learning as a new AI paradigm."
            ]
        }
    ]

    subject_tasks_map = {
        "foundations": [],
        "neural_nets": [],
        "distributed": [],
        "deep_learning": []
    }

    print(f"\n--- Populating {len(papers)} Papers into Notion ---")

    for i, paper in enumerate(papers, 1):
        subj_key = paper["subject_key"]
        subj_id = SUBJECT_MAP[subj_key]
        print(f"\n[{i}/10] Processing: {paper['short_title']}")

        # 1. Create Resource Entry with rich child blocks
        blocks = create_rich_block_structure(paper)

        resource_props = {
            "Resource Name": {
                "title": [{"text": {"content": paper["short_title"]}}]
            },
            "Type": {
                "select": {"name": "Article"}
            },
            "URL": {
                "url": paper["url"]
            },
            "Subjects": {
                "relation": [{"id": subj_id}]
            }
        }

        try:
            res_page = client.pages.create(
                parent={"database_id": resources_db_id},
                properties=resource_props,
                children=blocks
            )
            res_id = res_page.get("id")
            print(f"   [+] Created Resource Page: {res_id}")
        except Exception as e:
            print(f"   [!] Failed to create resource page: {e}")

        # 2. Create Task in Tasks Tracker
        task_props = {
            "Task name": {
                "title": [{"text": {"content": paper["task_title"]}}]
            },
            "Status": {
                "status": {"name": "Not started"}
            },
            "Tags": {
                "multi_select": [{"name": "Learning"}]
            },
            "Priority": {
                "select": {"name": paper["priority"]}
            },
            "Effort level": {
                "select": {"name": paper["effort"]}
            },
            "Description": {
                "rich_text": [{"text": {"content": paper["task_desc"]}}]
            }
        }

        try:
            task_page = client.pages.create(
                parent={"database_id": tasks_db_id},
                properties=task_props
            )
            task_id = task_page.get("id")
            print(f"   [+] Created Task: {task_id}")
            subject_tasks_map[subj_key].append(task_id)
        except Exception as e:
            print(f"   [!] Failed to create task: {e}")

    # 3. Link all Tasks back to their respective Subject pages
    print("\n--- Linking Tasks to Subjects ---")
    for subj_key, task_ids in subject_tasks_map.items():
        subj_id = SUBJECT_MAP[subj_key]
        if not task_ids:
            continue
        try:
            rel_array = [{"id": tid} for tid in task_ids]
            client.pages.update(
                page_id=subj_id,
                properties={"Tasks": {"relation": rel_array}}
            )
            print(f"   [+] Linked {len(task_ids)} tasks to Subject ({subj_key}): {subj_id}")
        except Exception as e:
            print(f"   [!] Failed to update subject {subj_key} tasks relation: {e}")

    print("\n🎉 ALL 10 PAPERS, RESOURCES, EMBEDS, AND TASKS HAVE BEEN SUCCESSFULLY CREATED IN NOTION!")

if __name__ == "__main__":
    main()
