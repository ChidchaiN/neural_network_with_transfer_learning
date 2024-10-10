import pandas as pd
import numpy as np
from random import random, seed
from math import exp
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import json

# Load data from Excel file
def load_excel(filename):
    try:
        df = pd.read_excel(filename, engine='openpyxl')
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

# Preprocess the data
def preprocess_data(data):
    X = data.iloc[:, :-1]  # Assuming last column is the target
    y = data.iloc[:, -1]   # Last column is the target

    # Convert categorical data to numerical
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_encoded, y_encoded

# Initialize a neural network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
    network.append(output_layer)
    return network

# Calculate neuron activation
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input through the network
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Backpropagate error
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = sum([neuron['weights'][j] * neuron['delta'] for neuron in network[i + 1]])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * neuron['output'] * (1.0 - neuron['output'])

# Update network weights
def update_weights(network, row, l_rate, freeze_layers):
    for i in range(len(network)):
        if freeze_layers and i < len(network) - 1:
            continue  # Skip weight updates for frozen layers
        inputs = row[:-1] if i == 0 else [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']  # Update bias

# Train the network
def train_network(network, train_data, l_rate, n_epoch, n_outputs, freeze_layers=False):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train_data:
            outputs = forward_propagate(network, row[:-1])  # Exclude target
            expected = [0 for _ in range(n_outputs)]
            expected[int(row[-1])] = 1  # Convert target to int
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate, freeze_layers)
        if epoch % 1000 == 0:  # Print error every 1000 epochs
            print(f"Epoch={epoch}, Learning Rate={l_rate}, Error={sum_error:.3f}")

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Global variable to hold training data
training_data = pd.DataFrame(columns=["Hair", "Eye", "Weight", "Lotion", "Sunburned"])

def load_training_data():
    global training_data
    try:
        training_data = pd.read_excel('sunburned.xlsx', engine='openpyxl')  # Load from Excel file
        print("Training data loaded successfully.")
    except FileNotFoundError:
        print("Training data file not found.")

def save_training_data():
    global training_data
    training_data.to_excel('sunburned.xlsx', index=False)  # Save to Excel file
    print("Training data saved successfully.")

def display_training_data():
    global training_data
    print("Displaying training data:")
    print(training_data)

def capitalize_first_letter(string):
    if isinstance(string, str) and len(string) > 0:
        return string[0].upper() + string[1:].lower()
    return string

def add_training_data():
    global training_data

    # Get user input
    hair = input("Enter Hair color: ")
    eye = input("Enter Eye color: ")
    weight = input("Enter Weight category: ")
    lotion = input("Is lotion applied? (Yes/No): ").strip().lower()
    sunburned = input("Did the individual get sunburned? (Yes/No): ").strip().lower()

    # Capitalize first letter of the inputs
    hair = capitalize_first_letter(hair)
    eye = capitalize_first_letter(eye)
    weight = capitalize_first_letter(weight)
    lotion = capitalize_first_letter(lotion)
    sunburned = capitalize_first_letter(sunburned)

    # Create a new DataFrame from user input
    new_data = pd.DataFrame({
        "Hair": [hair],
        "Eye": [eye],
        "Weight": [weight],
        "Lotion": [lotion],
        "Sunburned": [sunburned]
    })

    # Append the new data to the training data using pd.concat
    training_data = pd.concat([training_data, new_data], ignore_index=True)

    print("New training data added successfully.")

def edit_training_data(index):
    global training_data
    
    if index < 0 or index >= len(training_data):
        print("Invalid index. No changes made.")
        return

    print("Editing entry:", training_data.iloc[index])

    # Get user input for each field
    hair = input("Enter new Hair color (leave blank to keep current): ")
    eye = input("Enter new Eye color (leave blank to keep current): ")
    weight = input("Enter new Weight category (leave blank to keep current): ")
    lotion = input("Is lotion applied? (Yes/No, leave blank to keep current): ").strip().lower()
    sunburned = input("Did the individual get sunburned? (Yes/No, leave blank to keep current): ").strip().lower()

    # Update the fields only if new values are provided
    if hair:
        training_data.at[index, "Hair"] = capitalize_first_letter(hair)
    if eye:
        training_data.at[index, "Eye"] = capitalize_first_letter(eye)
    if weight:
        training_data.at[index, "Weight"] = capitalize_first_letter(weight)
    if lotion:
        training_data.at[index, "Lotion"] = capitalize_first_letter(lotion)
    if sunburned:
        training_data.at[index, "Sunburned"] = capitalize_first_letter(sunburned)

    print("Training data updated successfully.")

def search_training_data(value):
    global training_data  # Use the global training_data DataFrame

    # Convert input value to string for comparison
    value = str(value)

    # Check if the value exists in the index
    index_search = training_data.index.astype(str).str.contains(value)

    # Filter rows based on index match
    index_results = training_data[index_search]

    # Filter rows where any column contains the search value
    results = training_data[training_data.apply(lambda row: row.astype(str).str.contains(value, case=False).any(), axis=1)]

    # Combine the index search results with the column search results
    combined_results = pd.concat([index_results, results]).drop_duplicates()

    # Display results
    if combined_results.empty:
        print("No results found.")
    else:
        print("Search results:")
        print(combined_results)

def delete_training_data():
    global training_data
    index = int(input("Enter the index of the data to delete: "))
    if 0 <= index < len(training_data):
        training_data = training_data.drop(index).reset_index(drop=True)
        print("Training data deleted.")
    else:
        print("Invalid index.")

network = None

def train_model():
    global training_data, network  # Declare network as global
    # Preprocess the training data
    X, y = preprocess_data(training_data)

    # Initialize parameters
    n_inputs = X.shape[1]  # This should match the number of features
    n_outputs = len(set(y))

    # Get user input for number of hidden neurons and epochs
    n_hidden = int(input("Enter the number of hidden neurons: "))
    n_epoch = int(input("Enter the number of epochs: "))
    l_rate = 0.1

    if 'network' in globals() and network is not None:
        # Only re-initialize the output layer
        network[-1] = [{'weights': [random() for _ in range(len(network[-2]) + 1)]} for _ in range(n_outputs)]
    else:
        # Initialize entire network
        network = initialize_network(n_inputs, n_hidden, n_outputs)

    train_network(network, np.hstack((X, y.reshape(-1, 1))), l_rate, n_epoch, n_outputs)
    print("Model training completed.")


# Global variables to hold test results
predictions = []
actuals = []
accuracy = 0.0

def test_model():
    global training_data, network, predictions, actuals, accuracy  # Declare globals

    # Load the model first
    network = load_neuron()

    if network is None:
        print("Error: Model has not been trained yet. Please train the model first.")
        return

    if training_data is None or len(training_data) == 0:
        print("Error: Training data is not loaded. Please load the training data first.")
        return

    # Load test data from Excel file
    test_file = 'synthetic_data.xlsx'  # Ensure this file has the same features as the training data
    test_data = load_excel(test_file)

    # Preprocess the training data first to fit the encoder
    X_train, y_train = preprocess_data(training_data)

    # Preprocess the test data using the same encoder
    x_test, y_test = preprocess_data(test_data)

    # Map for predictions
    label_map = {0: "No", 1: "Yes"}

    # Initialize counters for results
    correct_predictions = 0
    total_predictions = len(y_test)

    predictions = []
    actuals = y_test.tolist()  # Store actual values

    # Convert actuals to strings if they are integers
    actuals_str = [label_map[int(y)] for y in actuals]  # Map integers to their string equivalents

    # Predict for each sample in the test data
    for i, sample_row in enumerate(x_test):
        sample_row = np.array(sample_row).flatten()  # Flatten if needed
        if sample_row.size != X_train.shape[1]:
            print(f"Sample {i + 1} has incorrect input size: {sample_row.size}. Expected: {X_train.shape[1]}.")
            continue  # Skip this sample if it doesn't match the expected size

        prediction = predict(network, sample_row)
        prediction_label = label_map[prediction]
        predictions.append(prediction_label)

        # Compare prediction with the actual label (using the mapped string version)
        if prediction_label == actuals_str[i]:
            correct_predictions += 1
        print(f"Sample {i + 1}: Predicted: {prediction_label}, Actual: {actuals_str[i]}")

    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    display_results(predictions, actuals_str, accuracy)  # Pass the string version for display
    
def display_results(predictions, actuals, accuracy):
    print("\n--- Test Results Summary ---")
    print(f"Total Samples: {len(actuals)}")
    print(f"Correct Predictions: {sum(pred == actual for pred, actual in zip(predictions, actuals))}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Optional: Create a confusion matrix representation
    confusion_matrix = pd.crosstab(pd.Series(predictions, name='Predicted'), pd.Series(actuals, name='Actual'))
    print("\nConfusion Matrix:")
    print(confusion_matrix)

    # Display percentages for better understanding
    total_samples = len(actuals)
    if total_samples > 0:
        no_predictions = predictions.count("No")
        yes_predictions = predictions.count("Yes")
        print(f"\nPredicted 'No': {100 * no_predictions / total_samples:.2f}%")
        print(f"Predicted 'Yes': {100 * yes_predictions / total_samples:.2f}%")

def save_neuron(network, filename='neural_network_model.json'):
    network_data = []
    for layer in network:
        layer_data = {
            'weights': [neuron['weights'] for neuron in layer]
        }
        network_data.append(layer_data)
    
    with open(filename, 'w') as f:
        json.dump(network_data, f)

def load_neuron(filename='neural_network_model.json'):
    global network
    with open(filename, 'r') as f:
        network_data = json.load(f)

    # Reconstruct the network from the saved data
    network = []
    for layer_data in network_data:
        layer = [{'weights': weights} for weights in layer_data['weights']]
        network.append(layer)

    print("Model loaded successfully.")
    return network

def main():
    while True:
        print("\nChoose an option:")
        print("1. Load training data")
        print("2. Save training data")
        print("3. Display training data")
        print("4. Add training data")
        print("5. Edit training data")
        print("6. Search training data")
        print("7. Delete training data")
        print("8. Train model")
        print("9. Display parameters from training")
        print("10. Test model")
        print("11. Save model")
        print("12. Load model")
        print("13. Exit")

        choice = input("Enter your choice (1-13): ")

        if choice == '1':
            load_training_data()
        elif choice == '2':
            save_training_data()
        elif choice == '3':
            display_training_data()
        elif choice == '4':
            add_training_data()
        elif choice == '5':
            index = int(input("Enter the index of the data to edit: "))
            edit_training_data(index)
        elif choice == '6':
            value = input("Enter the value to search for: ")
            search_training_data(value)
        elif choice == '7':
            delete_training_data()
        elif choice == '8':
            train_model()
        elif choice == '9':
            if predictions and actuals:
                display_results(predictions, actuals, accuracy)
            else:
                print("No results to display. Please run the test model first.")
        elif choice == '10':
            test_model()
        elif choice == '11':
            save_neuron(network)
        elif choice == '12':
            load_neuron()
        elif choice == '13':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
