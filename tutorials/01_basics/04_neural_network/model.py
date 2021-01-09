from constants import *
from data import train_loader, test_loader
from arch import NeuralNet

model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch_id in trange(1, NUM_EPOCHES + 1):
    epoch_loss = 0
    for batch_id, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        if images.size(0) != BATCH_SIZE: break #ignore last batch is its length is odd

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
    images = images.reshape(-1, 28 * 28).to(device)
    labels = labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total_output += labels.size(0)
    correct_output += (predicted == labels).sum()

print('Accuracy on testing images: {} %'.format(100 * correct_output / total_output))

# Save the model checkpoint
ckpt_path = CKPT_PATH + MODEL_NAME + ".ckpt"
print ("Checkpoint is saved at {}".format(ckpt_path))
torch.save(model.state_dict(), ckpt_path)