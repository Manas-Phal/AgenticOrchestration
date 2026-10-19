"""Custom evaluation dataset containing 20+ examples for DSPy optimization and evaluation."""

import dspy

# 20+ Curated Technical Research & Critic Evaluation Examples
RAW_DATASET_EXAMPLES = [
    {
        "question": "What are the core differences between Raft and Paxos consensus protocols?",
        "expected_topic": "Distributed Consensus",
        "expected_keywords": ["raft", "paxos", "leader election", "state machine replication"],
        "gold_summary": (
            "Raft decomposes consensus into leader election, log replication, and safety, "
            "whereas Paxos uses proposal numbers and two-phase commits."
        ),
    },
    {
        "question": "How does HNSW indexing work in vector databases?",
        "expected_topic": "Vector Search",
        "expected_keywords": ["hnsw", "graph", "hierarchical", "nearest neighbor", "similarity"],
        "gold_summary": (
            "Hierarchical Navigable Small World (HNSW) constructs multi-layer proximity graphs "
            "for approximate nearest neighbor vector search with logarithmic search scaling."
        ),
    },
    {
        "question": "What is the role of state checkpointing in LangGraph workflows?",
        "expected_topic": "Agentic Workflows",
        "expected_keywords": ["checkpointing", "langgraph", "state", "thread_id", "interrupt"],
        "gold_summary": (
            "State checkpointing persists graph state at each step using checkpointers like "
            "MemorySaver, enabling human-in-the-loop interrupts and fault tolerance."
        ),
    },
    {
        "question": "Explain Pydantic v2 performance improvements over v1.",
        "expected_topic": "Data Validation",
        "expected_keywords": ["pydantic", "pydantic-core", "rust", "validation", "performance"],
        "gold_summary": (
            "Pydantic v2 rewrote core validation logic in Rust (pydantic-core), achieving "
            "5-20x performance speedups and cleaner schema separation."
        ),
    },
    {
        "question": "What is the difference between Event Sourcing and CQRS?",
        "expected_topic": "Software Architecture",
        "expected_keywords": ["event sourcing", "cqrs", "command", "query", "audit log"],
        "gold_summary": (
            "Event Sourcing persists state changes as an append-only event sequence, while CQRS "
            "separates write command models from read query models."
        ),
    },
    {
        "question": "Compare LSM-Tree and B-Tree storage engines for write-heavy workloads.",
        "expected_topic": "Database Storage",
        "expected_keywords": ["lsm-tree", "b-tree", "write amplification", "sstables", "memtable"],
        "gold_summary": (
            "LSM-Trees buffer writes in a MemTable before sequential SSTable flushes, optimizing "
            "write throughput over B-Tree in-place updates."
        ),
    },
    {
        "question": "How does DSPy compile prompts declarative optimization?",
        "expected_topic": "Prompt Optimization",
        "expected_keywords": ["dspy", "signature", "teleprompter", "optimizer", "bootstrap"],
        "gold_summary": (
            "DSPy separates program logic (Signatures/Modules) from prompting strategies, using "
            "optimizers to compile few-shot examples and instructions automatically."
        ),
    },
    {
        "question": "What is OpenTelemetry distributed tracing and span context propagation?",
        "expected_topic": "Observability",
        "expected_keywords": ["opentelemetry", "tracing", "span", "trace_id", "context"],
        "gold_summary": (
            "OpenTelemetry traces microservice calls by injecting trace IDs and span IDs into "
            "request headers to track latency across service boundaries."
        ),
    },
    {
        "question": "How does rate limiting work with the Token Bucket algorithm?",
        "expected_topic": "API Design",
        "expected_keywords": ["rate limiting", "token bucket", "burst", "refill rate", "capacity"],
        "gold_summary": (
            "Token Bucket adds tokens to a bucket at a fixed rate; incoming requests consume "
            "tokens, allowing burst traffic while enforcing average rate limits."
        ),
    },
    {
        "question": "What is Write-Ahead Log (WAL) in relational databases like PostgreSQL?",
        "expected_topic": "Database Systems",
        "expected_keywords": ["wal", "postgresql", "durability", "acid", "crash recovery"],
        "gold_summary": (
            "Write-Ahead Logging records transaction changes to disk before updating data "
            "pages, ensuring ACID durability and crash recovery capability."
        ),
    },
    {
        "question": "How do circuit breakers prevent cascading failures in microservices?",
        "expected_topic": "Resilience Engineering",
        "expected_keywords": ["circuit breaker", "fallback", "open", "closed", "half-open"],
        "gold_summary": (
            "Circuit breakers monitor service failure rates, opening to trip call failures "
            "immediately without overwhelming failing downstream services."
        ),
    },
    {
        "question": "What is Model Context Protocol (MCP) tool integration?",
        "expected_topic": "AI Integration",
        "expected_keywords": ["mcp", "protocol", "tools", "resources", "json-rpc"],
        "gold_summary": (
            "MCP provides a standardized JSON-RPC protocol allowing AI clients to discover and "
            "invoke external tools and resources securely."
        ),
    },
    {
        "question": "Explain Consistent Hashing and its application in distributed caching.",
        "expected_topic": "Distributed Caching",
        "expected_keywords": ["consistent hashing", "hash ring", "virtual nodes", "rebalancing"],
        "gold_summary": (
            "Consistent Hashing maps keys and nodes onto a virtual ring, minimizing key "
            "relocation when nodes are added or removed from the cluster."
        ),
    },
    {
        "question": "What is Product Quantization (PQ) in vector compression?",
        "expected_topic": "Vector Compression",
        "expected_keywords": [
            "product quantization",
            "pq",
            "codebook",
            "subvectors",
            "compression",
        ],
        "gold_summary": (
            "Product Quantization splits high-dimensional vectors into subvectors, quantizing "
            "each to codebook centroids to drastically reduce memory usage."
        ),
    },
    {
        "question": "Compare gRPC and REST APIs for inter-service microservice communication.",
        "expected_topic": "API Protocols",
        "expected_keywords": ["grpc", "protobuf", "http/2", "rest", "json"],
        "gold_summary": (
            "gRPC uses HTTP/2 multiplexing and binary Protocol Buffers for high-performance RPCs, "
            "outperforming REST JSON serialization in microservices."
        ),
    },
    {
        "question": "How does Database Sharding achieve horizontal scaling?",
        "expected_topic": "Database Architecture",
        "expected_keywords": ["sharding", "shard key", "horizontal scaling", "partitioning"],
        "gold_summary": (
            "Sharding partitions data across independent database servers based on a shard key, "
            "enabling horizontal read/write scale-out."
        ),
    },
    {
        "question": "What is the function of PyTorch Autograd in gradient computation?",
        "expected_topic": "Machine Learning",
        "expected_keywords": [
            "pytorch",
            "autograd",
            "computational graph",
            "backpropagation",
            "gradient",
        ],
        "gold_summary": (
            "PyTorch Autograd records operations on tensors in a dynamic computational graph to "
            "execute automatic backpropagation for gradient optimization."
        ),
    },
    {
        "question": "Explain the zero-copy buffer optimization in high-performance networking.",
        "expected_topic": "Systems Performance",
        "expected_keywords": ["zero-copy", "sendfile", "dma", "kernel space", "user space"],
        "gold_summary": (
            "Zero-copy avoids CPU data duplication between kernel and user space buffers by "
            "transferring data directly via DMA hardware engines."
        ),
    },
    {
        "question": "What is the Difference Engine pattern in Retrieval-Augmented Generation?",
        "expected_topic": "RAG Engineering",
        "expected_keywords": ["rag", "retrieval", "reranking", "context compression", "synthesis"],
        "gold_summary": (
            "RAG architectures query vector indices, re-rank retrieved passages for relevance, "
            "and inject contextual snippets into LLM prompt contexts."
        ),
    },
    {
        "question": "How does Garbage Collection in Python handle circular references?",
        "expected_topic": "Python Internals",
        "expected_keywords": [
            "garbage collection",
            "reference counting",
            "gc",
            "cyclic trash",
            "generations",
        ],
        "gold_summary": (
            "Python uses reference counting for immediate cleanup and a generational cyclic "
            "garbage collector to detect and free unreachable object cycles."
        ),
    },
]


def load_dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load and format custom evaluation dataset into trainset (12) and devset (8).

    Returns:
        tuple containing (trainset, devset) list of dspy.Example objects.
    """
    dataset_examples = []
    for item in RAW_DATASET_EXAMPLES:
        ex = dspy.Example(
            question=item["question"],
            expected_topic=item["expected_topic"],
            expected_keywords=item["expected_keywords"],
            gold_summary=item["gold_summary"],
        ).with_inputs("question")
        dataset_examples.append(ex)

    trainset = dataset_examples[:12]
    devset = dataset_examples[12:]
    return trainset, devset
