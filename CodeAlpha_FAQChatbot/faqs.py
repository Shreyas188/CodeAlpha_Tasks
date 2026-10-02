"""
faqs.py - FAQ Data Module for CodeAlpha FAQ Chatbot
Contains 40+ Q&A pairs across multiple categories.
"""

FAQS = [
    # Python Basics
    {
        "question": "What is Python?",
        "answer": "Python is a high-level, interpreted, general-purpose programming language known for its simplicity, readability, and versatility. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "category": "Python Basics"
    },
    {
        "question": "What are Python data types?",
        "answer": "Python has several built-in data types: int (integers), float (decimal numbers), str (strings), bool (True/False), list (ordered mutable collection), tuple (ordered immutable collection), dict (key-value pairs), and set (unique unordered collection).",
        "category": "Python Basics"
    },
    {
        "question": "What is a Python list?",
        "answer": "A Python list is an ordered, mutable (changeable) collection of items enclosed in square brackets []. Lists can contain elements of different data types and support operations like append(), remove(), sort(), and slicing.",
        "category": "Python Basics"
    },
    {
        "question": "What is a Python dictionary?",
        "answer": "A Python dictionary is an unordered collection of key-value pairs enclosed in curly braces {}. Keys must be unique and immutable. Dictionaries allow fast data retrieval by key and support operations like get(), keys(), values(), and items().",
        "category": "Python Basics"
    },
    {
        "question": "What is a Python tuple?",
        "answer": "A Python tuple is an ordered, immutable collection of items enclosed in parentheses (). Unlike lists, tuples cannot be modified after creation. They are faster than lists and used for fixed data like coordinates or database records.",
        "category": "Python Basics"
    },
    {
        "question": "What is list comprehension?",
        "answer": "List comprehension is a concise way to create lists in Python. Syntax: [expression for item in iterable if condition]. Example: squares = [x**2 for x in range(10)] creates a list of squares from 0 to 81.",
        "category": "Python Basics"
    },
    {
        "question": "What is a lambda function?",
        "answer": "A lambda function is a small, anonymous function defined with the lambda keyword. Syntax: lambda arguments: expression. Example: square = lambda x: x**2. Lambda functions are useful for short, one-time use functions.",
        "category": "Python Basics"
    },
    {
        "question": "What is the difference between list and tuple?",
        "answer": "Lists are mutable (can be changed), use square brackets [], and are slower. Tuples are immutable (cannot be changed), use parentheses (), and are faster. Use lists for data that changes, tuples for fixed data.",
        "category": "Python Basics"
    },

    # OOP
    {
        "question": "What is Object Oriented Programming?",
        "answer": "Object-Oriented Programming (OOP) is a programming paradigm based on objects and classes. Key concepts include: Encapsulation (bundling data and methods), Inheritance (class hierarchy), Polymorphism (same interface, different behavior), and Abstraction (hiding complexity).",
        "category": "OOP"
    },
    {
        "question": "What is a class in Python?",
        "answer": "A class in Python is a blueprint for creating objects. It defines attributes (data) and methods (functions) that the objects will have. Use the 'class' keyword to define a class. Example: class Car: def __init__(self, brand): self.brand = brand",
        "category": "OOP"
    },
    {
        "question": "What is inheritance in Python?",
        "answer": "Inheritance allows a class (child) to inherit attributes and methods from another class (parent). This promotes code reuse. Use: class Child(Parent). Python supports single, multiple, multilevel, and hierarchical inheritance.",
        "category": "OOP"
    },
    {
        "question": "What is polymorphism?",
        "answer": "Polymorphism means the same interface can be used for different underlying forms. In Python, it allows different classes to have methods with the same name but different implementations. Achieved through method overriding and duck typing.",
        "category": "OOP"
    },
    {
        "question": "What is encapsulation?",
        "answer": "Encapsulation is binding data and methods together within a class, restricting direct access to some components. In Python, use single underscore _var (protected) or double underscore __var (private) to indicate access levels.",
        "category": "OOP"
    },

    # Machine Learning
    {
        "question": "What is machine learning?",
        "answer": "Machine Learning is a subset of AI where algorithms learn patterns from data to make predictions without being explicitly programmed. Types include: Supervised Learning (labeled data), Unsupervised Learning (unlabeled data), and Reinforcement Learning (reward-based).",
        "category": "Machine Learning"
    },
    {
        "question": "What is supervised learning?",
        "answer": "Supervised learning trains models on labeled data (input-output pairs). The model learns to map inputs to outputs. Examples: Linear Regression (predict prices), Logistic Regression (classify emails), Decision Trees, Random Forest, SVM.",
        "category": "Machine Learning"
    },
    {
        "question": "What is unsupervised learning?",
        "answer": "Unsupervised learning finds patterns in unlabeled data without predefined outputs. Examples: K-Means Clustering (group similar items), PCA (dimensionality reduction), Autoencoders. Used for customer segmentation, anomaly detection.",
        "category": "Machine Learning"
    },
    {
        "question": "What is deep learning?",
        "answer": "Deep Learning is a subset of ML using artificial neural networks with multiple layers. It excels at complex tasks like image recognition, NLP, and speech recognition. Frameworks: TensorFlow, PyTorch, Keras.",
        "category": "Machine Learning"
    },
    {
        "question": "What is overfitting?",
        "answer": "Overfitting occurs when a model learns training data too well, including noise, and performs poorly on new data. Solutions: More training data, Dropout, Regularization (L1/L2), Cross-validation, Early stopping, Simpler model.",
        "category": "Machine Learning"
    },
    {
        "question": "What is a neural network?",
        "answer": "A neural network is a computational model inspired by the human brain, consisting of layers of interconnected nodes (neurons). Layers: Input layer, Hidden layers, Output layer. Each connection has weights updated during training via backpropagation.",
        "category": "Machine Learning"
    },
    {
        "question": "What is the difference between AI and ML?",
        "answer": "AI (Artificial Intelligence) is the broad concept of machines simulating human intelligence. ML (Machine Learning) is a subset of AI where machines learn from data. Deep Learning is a subset of ML using neural networks. AI ⊃ ML ⊃ Deep Learning.",
        "category": "Machine Learning"
    },
    {
        "question": "What is NLP?",
        "answer": "Natural Language Processing (NLP) is AI for understanding and processing human language. Tasks: text classification, sentiment analysis, translation, named entity recognition, question answering. Libraries: NLTK, SpaCy, Hugging Face Transformers.",
        "category": "Machine Learning"
    },
    {
        "question": "What is TensorFlow?",
        "answer": "TensorFlow is an open-source ML framework by Google for building and training deep learning models. It supports neural networks, computer vision, NLP. Key API: tf.keras for high-level model building. Supports GPU acceleration.",
        "category": "Machine Learning"
    },
    {
        "question": "What is PyTorch?",
        "answer": "PyTorch is an open-source ML framework by Facebook/Meta, popular in research. Features: dynamic computation graphs, automatic differentiation, GPU support. Used for computer vision (torchvision), NLP (transformers), and custom architectures.",
        "category": "Machine Learning"
    },
    {
        "question": "What is scikit-learn?",
        "answer": "Scikit-learn is a Python ML library for classical ML algorithms. Features: classification, regression, clustering, dimensionality reduction, model selection, preprocessing. Easy API: model.fit(X_train, y_train), model.predict(X_test).",
        "category": "Machine Learning"
    },
    {
        "question": "What is a confusion matrix?",
        "answer": "A confusion matrix shows classification model performance. It displays TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative). From it: Accuracy=(TP+TN)/Total, Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2*P*R/(P+R).",
        "category": "Machine Learning"
    },
    {
        "question": "What is cross validation?",
        "answer": "Cross-validation is a technique to evaluate ML models by splitting data into k folds. Each fold acts as test set once while others train. K-Fold CV gives more reliable performance estimate than single train-test split. Use sklearn.model_selection.cross_val_score.",
        "category": "Machine Learning"
    },
    {
        "question": "What is transfer learning?",
        "answer": "Transfer learning uses a pre-trained model (trained on large dataset) as starting point for a new task. Benefits: less training data needed, faster training, better performance. Common in CV (VGG, ResNet) and NLP (BERT, GPT).",
        "category": "Machine Learning"
    },
    {
        "question": "What is a random forest?",
        "answer": "Random Forest is an ensemble ML algorithm that creates multiple decision trees and combines their predictions. Benefits: reduces overfitting, handles missing data, provides feature importance. Use sklearn.ensemble.RandomForestClassifier.",
        "category": "Machine Learning"
    },
    {
        "question": "What is a support vector machine?",
        "answer": "SVM (Support Vector Machine) is an ML algorithm that finds the optimal hyperplane to separate classes. Works well for high-dimensional data. Kernel trick allows non-linear classification. Use sklearn.svm.SVC for classification.",
        "category": "Machine Learning"
    },

    # Data Science
    {
        "question": "What is pandas?",
        "answer": "Pandas is a Python library for data manipulation and analysis. Key structures: DataFrame (2D table) and Series (1D array). Features: read CSV/Excel, filter data, group operations, merge datasets, handle missing values.",
        "category": "Data Science"
    },
    {
        "question": "What is numpy?",
        "answer": "NumPy is a Python library for numerical computing. It provides multi-dimensional arrays (ndarray), mathematical functions, linear algebra operations, and random number generation. It's the foundation for many scientific Python libraries.",
        "category": "Data Science"
    },
    {
        "question": "What is matplotlib?",
        "answer": "Matplotlib is a Python plotting library for creating static, animated, and interactive visualizations. Common plots: line, bar, scatter, histogram, pie. Use plt.plot(), plt.show(), plt.savefig() for basic plotting.",
        "category": "Data Science"
    },
    {
        "question": "What is data preprocessing?",
        "answer": "Data preprocessing prepares raw data for ML models. Steps: Handle missing values (imputation/removal), Encoding categorical variables, Feature scaling (normalization/standardization), Feature selection, Train-test splitting.",
        "category": "Data Science"
    },
    {
        "question": "What is feature engineering?",
        "answer": "Feature engineering is the process of creating or transforming input features to improve ML model performance. Techniques: feature extraction, polynomial features, binning, one-hot encoding, feature scaling, PCA for dimensionality reduction.",
        "category": "Data Science"
    },

    # Git/GitHub
    {
        "question": "What is Git?",
        "answer": "Git is a distributed version control system for tracking code changes, collaborating with teams, and managing project history. Key commands: git init, git add, git commit, git push, git pull, git clone, git branch, git merge.",
        "category": "Git/GitHub"
    },
    {
        "question": "What is GitHub?",
        "answer": "GitHub is a web-based platform for hosting Git repositories. Features: code hosting, collaboration, pull requests, issues tracking, GitHub Actions (CI/CD), GitHub Pages. It's the largest code hosting platform with 100M+ developers.",
        "category": "Git/GitHub"
    },
    {
        "question": "How to push code to GitHub?",
        "answer": "Steps to push to GitHub: 1) git init, 2) git add . (stage files), 3) git commit -m 'message', 4) git remote add origin <URL>, 5) git push -u origin main. Use git push for subsequent pushes.",
        "category": "Git/GitHub"
    },

    # General
    {
        "question": "What is an API?",
        "answer": "An API (Application Programming Interface) is a set of rules allowing different software applications to communicate. REST APIs use HTTP methods (GET, POST, PUT, DELETE). Use Python's requests library to call APIs: requests.get(url, params=params).",
        "category": "General"
    },
    {
        "question": "What is JSON?",
        "answer": "JSON (JavaScript Object Notation) is a lightweight data format for storing and transmitting data. It uses key-value pairs like Python dictionaries. Python's json module: json.loads() to parse, json.dumps() to serialize.",
        "category": "General"
    },
    {
        "question": "What is a virtual environment?",
        "answer": "A virtual environment is an isolated Python environment with its own packages. Create: python -m venv venv. Activate: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Mac/Linux). Install packages: pip install package.",
        "category": "General"
    },
]

# Intent keyword lists
GREETINGS = [
    "hello", "hi", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "greetings", "what's up", "sup"
]

FAREWELLS = [
    "bye", "goodbye", "see you", "see ya", "take care",
    "farewell", "quit", "exit", "later"
]

THANKS = [
    "thank you", "thanks", "thank u", "thx", "ty",
    "appreciate it", "great", "awesome", "wonderful", "helpful"
]
