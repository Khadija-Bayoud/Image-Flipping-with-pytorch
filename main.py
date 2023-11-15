import cv2
import torch
import numpy as np
from NeuralNet import NeuralNet

if __name__ == '__main__' : 
    
    # Original
    original = cv2.imread('shirohige.jpeg')
    
    # Flipped
    flipped = cv2.flip(original, 0)
    
    # Normalizing 
    original = original/255
    flipped = flipped/255
    print(f'Min : {original.min()} -- Max : {original.max()}')
    
    # Resizing
    original = cv2.resize(original, (500, 500), 0.75, 0.75)
    flipped = cv2.resize(flipped, (500, 500), 0.75, 0.75)
    
    # Flattening
    X = original.flatten().reshape(1, -1)
    Y = flipped.flatten().reshape(1, -1)
    print(f'Input shape : {X.shape}')
    
    # Converting np -> torch
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    
    # Window for real time visualization
    window_name = "Original and Predicted Images"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Neural Net
    input_size = X.shape[1]
    model = NeuralNet(input_size, 1, input_size).to("cpu")
    
    # Training
    MAX_ITER = 2000
    learning_rate = 1000
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for iter in range(MAX_ITER):
        optimizer.zero_grad()
        prediction = model.forward(X)
        loss = criterion(prediction, Y)
        loss.backward()
        print(f'Iter: {iter} --- Loss: {loss.item()}')
        optimizer.step()
        
        output = prediction.detach().numpy()
        output_image = output.reshape(original.shape)
        
        display_image = np.vstack((original, output_image))
        cv2.imshow(window_name, display_image) 
        cv2.waitKey(1)  
        
    while True:
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break  