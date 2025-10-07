# SARSA Linear Function Approximation for Mountain Car

This project aims to implement a **SARSA (State-Action-Reward-State-Action) agent** with **linear function approximation** for solving the Mountain Car environment from Gymnasium. The agent uses various feature extraction methods to convert continuous state spaces into feature vectors suitable for linear Q-value approximation.

## 🎯 **Key Features**

- **Modular Architecture**: Clean separation with abstract base classes
- **Model Persistence**: Save and load trained models

## 🏗️ **Architecture**

### **Core Components**

1. **`sarsa_linear_agent.py`** - Main SARSA agent implementation
2. **`feature_extractors.py`** - Abstract and concrete feature extraction classes
4. **`example.py`** - Usage examples and demonstrations

### **Feature Extractors**

#### **Radial Basis Functions** (`RBFFeatureExtractor`) 
- Uses Gaussian-like RBF centers distributed across state space
- **Best for**: Localized learning, non-linear patterns
- **Parameters**: `n_centers` (default: 25), `sigma` (default: 0.1)

## 🚀 **Quick Start**

### **Basic Usage**

```bash
# Train with RBF features (default)
python example.py
```


## 📊 **Analysis & Comparison**


## 🛠️ **Dependencies**

```bash
pip install gymnasium numpy matplotlib pytest
```

## 📁 **File Structure**

```
linear_mountain_car/
├── sarsa_linear_agent.py      # Core SARSA agent
├── feature_extractors.py      # Feature extraction implementations  
├── training.py                # Training, evaluation, CLI
├── example.py                 # Usage examples
├── feature_demo.py            # Feature extractor analysis
├── README.md                  # This documentation
└── requirements.txt           # Dependencies
```

## 🎯 **Algorithm Details**

### **SARSA Update Rule**
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```

Where:
- **α**: Learning rate
- **γ**: Discount factor  
- **r**: Immediate reward
- **s,a**: Current state-action pair
- **s',a'**: Next state-action pair

### **Linear Function Approximation**
```
Q(s,a) = θ^T φ(s,a)
```

Where:
- **θ**: Weight vector
- **φ(s,a)**: Feature vector for state-action pair

### **ε-Greedy Exploration**
```
π(s) = argmax_a Q(s,a) with probability (1-ε)
       random action    with probability ε
```


## 📖 **References**

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- Gymnasium Mountain Car Environment Documentation