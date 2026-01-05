## Effective Knowledge Transfer in Federated Learning with Non-IID Data via Feature Distribution Alignment

### About the research
In this study, we propose a general Federated Learning (FL) framework called Feature Distribution Alignment (FDA) to address the imbalance between global and local knowledge in Personalized Federated Learning (PFL). FDA enhances collaborative knowledge transfer by aligning global and local feature distributions, enabling personalized local models to effectively learn consensus knowledge while preserving client-specific features. This framework improves the effectiveness, stability, and fairness of FL under diverse non-IID conditions.

<img width="3144" height="1548" alt="framework" src="https://github.com/user-attachments/assets/47392b75-ef74-4a49-9547-1d45c7d6f9b2" />

To overcome the degradation caused by the periodic reinitialization of local models with global updates, FDA introduces two complementary components: Consensus Knowledge Guidance (CKG) and Local Knowledge Memory (LKM).
- **Consensus Knowledge Guidance (CKG)** adopts a domain adversarial mechanism adapted to FL environments. It guides local feature extractors to align feature dimensions with globally invariant semantics, thereby promoting consistent knowledge transfer across heterogeneous clients.
- **Local Knowledge Memory (LKM)** preserves historical personalized information through attention-based fusion and mutual information maximization, which jointly enhance feature alignment and minimize redundancy.

This dual mechanism ensures that local models continuously retain their prior knowledge while synchronizing with global representations, allowing both global generalization and local personalization to be effectively maintained throughout the training process.

The FDA framework operates in three key stages:
1. **Global Knowledge Sharing**: The server broadcasts the global model to clients after each aggregation step.
2. **Local Knowledge Retention**: Each client leverages LKM to fuse the newly received global features with historical local representations, maintaining personalized consistency.
3. **Collaborative Feature Alignment**: CKG enforces feature distribution alignment through domain adversarial learning, ensuring global-local semantic consistency across clients.

Theoretical analysis confirms that FDA improves server-to-client knowledge transfer by tightening the generalization bounds of both global and local feature extractors. Comprehensive experiments conducted on four datasets under three types of statistical heterogeneity demonstrate that FedAvg+FDA consistently outperforms state-of-the-art PFL methods in accuracy, robustness, and fairness.

For implementation details, model design, and experimental configurations, please refer to our paper entitled: “Effective Knowledge Transfer in Federated Learning with Non-IID Data via Feature Distribution Alignment.”

### Datasets and Environments
Due to the file size limitation, we only upload the fmnist dataset with the default practical setting (β=0.1). Please refer to our project [PFLlib](https://github.com/TsingZ0/PFLlib) for other datasets and environments settings.

### System
- `main.py`: configurations of `FedAvg+FDA`.  
- `env_linux.yaml`: python environment to run `FedAvg+FDA` on Linux.  
- `./flcore`:  
  - `./clients/clientDBE.py`: the code on the client.  
  - `./servers/serverDBE.py`: the code on the server.  
  - `./trainmodel/models.py`: the code for models.  
- `./utils`:  
  - `data_utils.py`: the code to read the dataset. 

### Training and Evaluation
All codes corresponding to **FedAvg+FDA** are stored in `./system`.  
Just run the following commands:


cd ./system
`python main.py`
