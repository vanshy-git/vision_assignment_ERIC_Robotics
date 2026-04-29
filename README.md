# Robotics Representation Learning (TUM RGB-D) - Vansh
## Contact 
* **Name:** Vansh
* **Contact Number:** +91 9389177186
* **Email Address:** vansh.y.v.y@gmail.com


This project implements a self-supervised visual perception pipeline for robotics. Using only **500 unlabeled frames**, I developed a model that learns the "geometry of a room" without human labels, achieving a **+16.2% accuracy lift** over standard supervised training.

---

## Pipeline
1.  **Data Engineering:** Sampled the `fr1/xyz` dataset to capture translational motion trajectories.
2.  **Contrastive Pretraining:** Implemented a SimCLR-style logic using a **ResNet-18** backbone. The model learned by comparing "distorted twins" of images to extract invariant features.
3.  **Dimensionality Reduction:** Squashed 512-D features into 2D using a **PCA + t-SNE** stack to verify cluster quality.
4.  **Downstream Evaluation:** Validated representations by classifying camera "zones" (Start/Middle/End).

---

## Bridging the "16% Accuracy Gap"
The most critical part of this assignment was comparing the **Pretrained Model (50%)** vs. the **Scratch Model (33.8%)**.

* **The Problem:** With only 500 labeled samples, the Scratch model is feature-blind. It cannot learn how to see and how to classify simultaneously, resulting in performance no better than a random guess.
* **The Solution:** My pretrained model uses **Feature Priming**. It spent 30 epochs learning edges, textures, and spatial layouts unsupervised. When the tiny labeled dataset was introduced, it already had a structured visual vocabulary, allowing it to immediately outperform the baseline by over 16%.

---

## Technical Details
* **t-SNE Trajectories:** The visualization shows clear "snake-like" paths. This proves the model successfully learned a manifold representing the camera's physical motion.
* **Augmentation Strategy:** I used a heavy mix of `RandomResizedCrop` and `ColorJitter`. This forced the CNN to ignore lighting noise and focus on the underlying geometric structure, a necessity for robust robotics.
* **Advanced Viz:** By using PCA as an initial step before t-SNE, I preserved global variance while revealing local semantic clusters, a best practice for high-dimensional feature analysis.

---

## Project Structure
* `dataset.py`: Dual-view augmentation logic.
* `model.py`: ResNet-18 backbone + Projection Head.
* `train.py`: Self-supervised pretraining engine.
* `visualize.py`: PCA & t-SNE embedding plots.
* `evaluate.py`: Pretrained vs. Scratch comparison script.