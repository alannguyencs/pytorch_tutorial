from constants import *


class Model:
    def __init__(self, arch):
        self.model = arch.to(device)
        self.ckpt_path = CKPT_PATH + self.model.name + ".ckpt"

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = None

        self.num_epochs = NUM_EPOCHS

        self.total = 0
        self.correct = 0
        self.epoch_loss = 0
        self.epoch_losses = None
        self.min_epoch_loss = INF
        self.final_model_state = None

    def train(self, train_loader):
        for _ in trange(1, self.num_epochs + 1):
            self.run_epoch(loader=train_loader, is_training=True)
            self.update_result()

        self.save_final_model_state()

    def test(self, test_loader):
        self.run_epoch(loader=test_loader, is_training=False)

    def run_epoch(self, loader, is_training):
        self.model.train() if is_training else self.model.eval()
        self.refresh_evaluation_metrics()
        for images, labels in loader:
            if images.size(0) != BATCH_SIZE: continue
            self.run_batch(images, labels)
        self.summarize_epoch()

    def run_batch(self, images, labels):
        images = images.to(device)
        labels = labels.to(device)

        outputs = self.model(images)
        self.loss = self.criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()
        self.epoch_losses.append(self.loss.item())

        if self.model.training: self.backward_and_optimize()

    def refresh_evaluation_metrics(self):
        self.total = 0
        self.correct = 0
        self.epoch_losses = []

    def backward_and_optimize(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def summarize_epoch(self):
        epoch_loss = sum(self.epoch_losses) / max(len(self.epoch_losses), 1)
        epoch_accuracy = self.correct / self.total * 100
        print ("Epoch loss = {:.6f}, accuracy = {:.2f}%".format(epoch_loss, epoch_accuracy))

    def update_result(self):
        if self.min_epoch_loss > self.epoch_loss:
            self.min_epoch_loss = self.epoch_loss
            self.final_model_state = self.model.state_dict()

    def save_final_model_state(self):
        if self.final_model_state is None:
            return

        torch.save(self.final_model_state, self.ckpt_path)
        print("final model state is saved at " + self.ckpt_path)
