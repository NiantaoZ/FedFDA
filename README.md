## Effective Knowledge Transfer in Federated Learning with Non-IID Data via Feature Distribution Alignment

### About the research
In this study, we propose a general Federated Learning (FL) framework called Feature Distribution Alignment (FDA) to address the imbalance between global and local knowledge in Personalized Federated Learning (PFL). FDA enhances collaborative knowledge transfer by aligning global and local feature distributions, enabling personalized local models to effectively learn consensus knowledge while preserving client-specific features. This framework improves the effectiveness, stability, and fairness of FL under diverse non-IID conditions.

<img width="3144" height="1548" alt="framework" src="https://github.com/user-attachments/assets/47392b75-ef74-4a49-9547-1d45c7d6f9b2" />


To overcome the degradation caused by the periodic reinitialization of local models with global updates, FDA introduces two complementary components: Consensus Knowledge Guidance (CKG) and Local Knowledge Memory (LKM).

Due to the file size limitation, we only upload the fmnist dataset with the default practical setting (β=0.1). Please refer to our project [PFLlib](https://github.com/TsingZ0/PFLlib) for other datasets and environments settings.
