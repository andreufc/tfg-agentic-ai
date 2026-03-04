# tfg-agentic-ai
Repository for the code of my final thesis of my GCED (Grau en Ciència i Enginyeria de Dades) bachelor's degree.

This thesis explores using multi-agent collaboration (a Society of Minds approach) to enhance the reasoning capabilities of Small Language Models (SLMs) like Phi-3 and TinyLlama, which often struggle with complex tasks. The goal is to develop an optimized pipeline where specialized agents collaboratively solve tasks in GSM8K and MMLU, aiming for accuracy gains over single-agent inference with minimal resources. Technically, the Society-of-Minds pipeline will use LangChain (LCEL) for compositional agent calls and LangGraph to represent each collaboration protocol as a stateful graph with unified tracing (tokens/latency) and W&B logging.

The code is ran through Colab notebooks using the provided T4 GPU with 16GB VRAM.
