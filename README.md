# Simplified Paper Implementations

## Project Goal

This project aims to provide simplified implementations of research papers. While many original authors release their code, it often includes complex distributed, parallel, or framework-specific code that can obscure the core innovations of the paper for readers.

This project addresses this challenge by simplifying the implementations based on the original papers and the authors' public code. All complex distributed, parallel, and framework-specific code has been removed to offer the most straightforward replications of the papers' core ideas.

## Implemented Papers

Currently, the following papers have been implemented:

1.  **`Absolute Zero Reinforced Self-play Reasoning with Zero Data.py`**: Corresponds to the paper "Absolute Zero Reinforced Self-play Reasoning with Zero Data" https://arxiv.org/abs/2505.03335
2.  **`LUFFY.py`**: Corresponds to the paper "Learning to Reason under OFF-Policy guidance"https://arxiv.org/abs/2504.14945
3.  **`GRPO.py`**: Based on the implementation found at [https://github.com/aburkov/theLMbook](https://github.com/aburkov/theLMbook).
4.  **`swin_transformer.py`**: A reproduction of the paper "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".
5.  **`vit.py`**: A reproduction of the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".
6.  **`reinforce++.py`**: A reproduction of "REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models". **Important Note:** I do not currently endorse this paper, as I believe it misinterprets the concept of advantage and provides insufficient justification for its batch normalization approach, with inadequate experimental support. It is recommended to await further community feedback before use.
7.  **`Qformer.py`**: A reproduction of the Qformer module.
8.  **`MA-LMM.py`**: A reproduction of the paper "MA-LMM Memory-Augmented Large Multimodal Model" [https://github.com/boheumd/MA-LMM](https://arxiv.org/abs/2404.05726).