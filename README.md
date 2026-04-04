# Asynchronous joint distribution alignment

## 🔍 Overview

Distribution alignment mechanism serves as a fundamental basis for a cross-domain fault diagnosis, directly affecting diagnostic accuracy. A typical joint distribution alignment which enables a fine-grained class-level subdomain alignment, has achieved notable success in some cases. However, this mechanism performs a synchronous class-wise alignment across all categories between the source and the target domains, without considering the confidence of pseudo-labels. This approach may lead to the collapse of the entire knowledge transfer framework due to an unreliable alignment. By contrast, an asynchronous joint distribution alignment (AJDA) mechanism is developed to enhance the model stability. In AJDA, the asynchronous class-level alignment strategy is first proposed to adaptively select high-confidence class-conditional distributions from the training process. Furthermore, a new high-order significant discrepancy representation metric based on the mechanical vibration characteristics is constructed to improve the sensitivity in measuring cross-domain distribution shifts. Finally, the proposed AJDA mechanism is validated through two scenarios, including planetary gearbox cross-condition and public cross-bearing transfer diagnosis cases, achieving average diagnostic accuracies of 98.81% and 80.23%, respectively. AJDA yields an overall performance improvement of more than 20% compared with other well-known methods, confirming its effectiveness and advantage.

## 📝 Citation

Quan Qian et al. Asynchronous joint distribution alignment: a new domain confusion mechanism for fault transfer diagnosis
IEEE/ASME Transactions on Mechatronics
