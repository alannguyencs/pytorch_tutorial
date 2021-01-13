from constants import *

class Model():
    def __init__(self, arch):
        self.model = arch.to(device)
        self.ckpt_path = CKPT_PATH + self.model.name + ".ckpt"

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = None

        self.num_epoches = NUM_EPOCHES

        self.total = 0
        self.correct = 0
        self.epoch_loss = 0
        self.min_epoch_loss = 1e6

    def train(self, train_loader):
        for _ in trange(1, self.num_epoches + 1):
            self.run_epoch(loader=train_loader, is_training=True)
            self.summary_epoch()
            self.save_checkpoint()
        print("checkpoint is saved at " + self.ckpt_path)

    def test(self, test_loader):
        self.run_epoch(loader=test_loader, is_training=False)
        self.summary_epoch()

    def run_epoch(self, loader, is_training):
        self.model.train() if is_training else self.model.eval()
        self.refresh_eval_and_loss()
        for images, labels in loader:
            if images.size(0) != BATCH_SIZE: continue
            self.run_batch(images, labels)
        self.epoch_loss /= len(loader)


    def run_batch(self, images, labels):
        images = images.to(device)
        labels = labels.to(device)

        outputs = self.model(images)
        self.loss = self.criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()
        self.epoch_loss += self.loss.item()

        self.backward_and_optimize()

    def refresh_eval_and_loss(self):
        self.total = 0
        self.correct = 0
        self.epoch_loss = 0

    def backward_and_optimize(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def summary_epoch(self):
        print ("Epoch loss = {:.6f}, Accuracy = {:.2f}%"
               .format(self.epoch_loss, self.correct / self.total * 100))

    def save_checkpoint(self):
        if self.min_epoch_loss > self.epoch_loss:
            self.min_epoch_loss = self.epoch_loss
            torch.save(self.model.state_dict(), self.ckpt_path)
