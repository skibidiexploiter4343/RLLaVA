<div align="center">
  <img src="assets/icon.png" alt="RLLaVA Icon" width="200">
</div>

# RLLaVA: A RL-central Framework for Language and Vision Assistant 🚀

> **Algorithm-Driven Design • Native Multimodal Support • Production-Ready Training**

## 💡 What is RLLaVA?

RLLaVA is a **flexible, production-ready framework** for applying reinforcement learning to vision-language models.

**Key Innovation:** Unlike traditional frameworks that intertwine algorithm logic with distributed system details, RLLaVA's **algorithm-driven architecture** lets you focus on RL research—not system engineering.

**Built for:** Researchers who want to experiment with diverse RL algorithms (GRPO, PPO, REMAX, etc.) on multimodal tasks—from math reasoning to agentic applications—without rewriting training infrastructure.

Built upon TinyLLaVA's design philosophy of lightweight and extensible architecture, RLLaVA provides a powerful yet accessible platform for multimodal RL development.

<div align="center">
  <img src="assets/arch.png" alt="RLLaVA Architecture" width="600">
</div>

---

## ✨ Core Features

### 🎯 Algorithm-Driven Design
- **Comprehensive RL Algorithm Support**: GRPO, RLOO, PPO, REMAX, DAPO, GMPO, and more—switch algorithms with minimal config changes
- **Hybrid Training Methods**: Native support for CHORD, LUFFY, UFT that unify SFT and RL in a single training loop
- **Plugin System**: Register custom algorithms with `@register_ppo("my_algo")` decorator
- **Decoupled Architecture**: Algorithm logic separated from distributed execution for maximum flexibility

### 🌈 Native Multi-Modal Support  
- **Unified Data Protocol**: Seamless handling of image, video, and text inputs—accessible in both training loops and reward functions
- **Validated Across Modalities**: Proven on math reasoning (Geometry3K), visual tasks (CLEVR, RefCOCO, COCO, LVIS), and agentic applications (MAT-Search, DeepEyes)—with easy extensibility for new tasks

### ⚡ Production-Ready Infrastructure
- **Familiar HuggingFace & PyTorch APIs**: Consistent design patterns and naming conventions—minimal learning curve for existing users
- **Distributed Training**: Multiple backends—FSDP, DeepSpeed, HuggingFace Accelerate
- **Efficient Inference**: Integrated with vLLM and SGLang for fast rollout generation
- **Memory Optimizations**: LoRA, gradient checkpointing, dynamic batching, padding-free training for resource-constrained environments

## 🚀 Quick Start

### Install
```bash
git clone https://github.com/TinyLoopX/RLLaVA && cd RLLaVA

conda create -n rllava python==3.11 && conda activate rllava

bash ./install.sh
```


## 🏗️ How It Works

RLLaVA simplifies RL training into **3 steps**:

**Step 1: Define your task** (reward function + prompt template)
```python
# examples/reward_function/my_task.py
def compute_score(response, ground_truth):
    return 1.0 if response == ground_truth else 0.0

# examples/format_prompt/my_task.jinja
{{ question }}\nAnswer: 
```

**Step 2: Configure training** (YAML file)
```yaml
data:
  train_files: your_org/your_dataset
  format_prompt: ./examples/format_prompt/my_task.jinja
reward:
  reward_function: ./examples/reward_function/my_task.py:compute_score
algorithm:
  adv_estimator: grpo  # or ppo, rloo, remax, etc.
```

**Step 3: Run**
```python
pipeline = RLVRPipeline(model, config, train_dataloader, val_dataloader)
pipeline.run()  # rollout → advantage → update → repeat
```

**That's it!** The framework handles distributed training, inference optimization, and memory management automatically.



## 📦 Examples

**Ready-to-run scripts** covering diverse algorithms and tasks:

```bash
examples/
├── algorithms/  # Algorithm comparisons (GRPO, RLOO, DAPO, etc.)
├── tasks/       # Task-specific configs (math, vision, agentic, hybrid training)
└── custom/      # Templates for reward functions and prompts
```

**🎯 Add Your Custom Task in 3 Files:**

1. **Reward function** → `examples/reward_function/your_task.py`
2. **Prompt template** → `examples/format_prompt/your_task.jinja`  
3. **Config YAML** → Point to dataset + reward + prompt:
   ```yaml
   data: 
     train_files: your_org/dataset@train
     format_prompt: ./examples/format_prompt/your_task.jinja
   reward: 
     reward_function: ./examples/reward_function/your_task.py:compute_score
   ```

## 🌈 Supported Models & Modalities

**Vision-Language Models:**
- Qwen2-VL, Qwen2.5-VL (2B, 3B, 7B variants)
- LLaVA-style architectures with flexible vision encoders (SigLIP, DINOv2)
- Extensible design for integrating new multimodal models

**Input Modalities:**
- **Images**: Dynamic resolution (262K-4M pixels), batch processing with padding optimization
- **Videos**: Initial support with plans for further extension
- **Text**: Standard language model inputs with multimodal context

## 🎯 Algorithms

**Comprehensive RL algorithms—switch with minimal config changes:**

```yaml
algorithm:
  adv_estimator: grpo  # Change advantage estimator
  # Some algorithms may require additional parameters
  # e.g., DAPO: policy_loss_fn: dapo, GSPO: policy_loss_fn: gspo
```

### Supported Algorithms

| Category | Algorithms |
|----------|-----------|
| **Policy Gradients** | REINFORCE++, GRPO, RLOO, OPO |
| **Value-Based** | GAE, PPO |
| **Variance Reduction** | REMAX, GPG, CLIP-COV, KL-COV |
| **Optimization Methods** | DAPO, GMPO, GSPO, DR-GRPO, GRPO-PassK |
| **Hybrid Training** | CHORD, LUFFY, UFT |

**Hybrid Training Methods:**
- **CHORD**: Expert-guided training with adaptive SFT/RL mixing ratios
- **LUFFY**: Off-policy learning for long-horizon response generation
- **UFT**: Unified framework combining supervised and reinforcement fine-tuning

**Add Your Custom Algorithm:**
```python
@register_ppo("my_algo")
class MyAlgo(PPO):
    def compute_advantage(self, batch):
        # Your advantage computation logic
        ...
```

*Continuous integration of emerging algorithms—contributions welcome!*

## 🔮 Roadmap

**Core focus areas:**
- **Multi-Modal Agentic Tasks**: Advanced agentic applications integrating vision, language, and action—web agents, coding assistants, tool-use orchestration with visual context
- **Embodied Intelligence**: RL training for vision-language-action models in simulated and real-world environments—manipulation, navigation, interactive decision-making
- **Video Understanding**: Temporal modeling and long-video processing for video-based reasoning and agentic tasks
- **Extended Model & Task Support**: Broader vision-language architectures and richer multi-modal task coverage

**Research directions:**
- Multi-agent collaboration in shared visual environments
- Efficient training for resource-constrained embodied settings
- Long-horizon planning and reasoning with visual feedback

*We welcome community feedback and collaboration opportunities in multi-modal agentic AI and embodied intelligence.*

## 🙏 Acknowledgements

Inspired by [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory)'s modular design and the broader RL community: [verl](https://github.com/volcengine/verl), ROLL, AReaL, RLite, RLinf, and many others.

## 🤝 Contributing

We welcome contributions! Here's how to get started:

**Quick Wins:**
- 🐛 [Report bugs or request features](https://github.com/TinyLoopX/RLLaVA/issues) - Help us improve
- 📝 Improve documentation or add examples - Share your experience
- 🧪 Share training results and configurations - Help the community

**High-Impact Areas:**
- 🧠 **New RL algorithms**: Extend `rllava/ppo/` with your algorithm implementation
- 🎯 **Multi-modal tasks**: Add reward functions for new domains (vision, agentic, embodied)
- ⚡ **Performance optimizations**: Memory efficiency, distributed training, inference speedup

**Current Focus:**
- Multi-modal agentic tasks and embodied intelligence
- Video understanding and temporal reasoning
- Cross-modal attention and reasoning mechanisms

See our [Contributing Guidelines](CONTRIBUTING.md) for detailed instructions.

## 📄 License

Apache-2.0 • Built with ❤️ for the research community
