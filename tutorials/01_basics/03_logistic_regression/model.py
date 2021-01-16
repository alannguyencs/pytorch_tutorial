from constants import *
from data import train_loader, test_loader

# Logistic regression model
model = nn.Linear(INPUT_SIZE, NUM_CLASSES)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() #negative log likelyhood
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch_id in range(1, NUM_EPOCHES + 1):
    epoch_loss = 0
    for batch_id, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28) # batch_size x 28 x 28 --> batch_size x 784

        # ignore last batch if loader_length % batch_size != 0
        if images.size(0) < BATCH_SIZE: break

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ('Epoch [{}/{}], Loss: {:.4f}'
                      .format(epoch_id, NUM_EPOCHES, epoch_loss / len(train_loader)))

# Test the model
model.eval()
correct_output, total_output = 0, 0
for images, labels in test_loader:
    images = images.reshape(-1, 28*28)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)  #outputs: batch_size x 10
    total_output += images.size(0)  # += batch_size
    correct_output += (predicted == labels).sum()

print('Accuracy on testing images: {} %'.format(100 * correct_output / total_output))

# Save the model checkpoint
ckpt_path = CKPT_PATH + MODEL_NAME + ".ckpt"
print ("Checkpoint is saved at {}".format(ckpt_path))
torch.save(model.state_dict(), ckpt_path)