from constants import *
import data


# Linear regression model
model = nn.Linear(INPUT_SIZE, TARGET_SIZE)


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


# Train the model
for epoch in range(1, NUM_EPOCHES+1):
    preds = model(data.inputs)
    loss = criterion(preds, data.targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, NUM_EPOCHES, loss.item()))

# Save the model checkpoint
ckpt_path = CKPT_PATH + MODEL_NAME + ".ckpt"
print ("Checkpoint is saved at {}".format(ckpt_path))
torch.save(model.state_dict(), ckpt_path)



#Visualize trained model
#Remark: one model includes two modes: train(), eval()
model.eval()  #change to evaluation mode
preds = model(data.inputs).data.numpy()
plt.plot(data.inputs.numpy(), data.targets.numpy(), 'ro', label='Original data')
plt.plot(data.inputs.numpy(), preds, label='Fitted line')
plt.legend()
plt.show()


