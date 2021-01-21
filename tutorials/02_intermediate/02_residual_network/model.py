from constants import *

class Model():
    def __init__(self, arch):
        self.model = arch.to(device)
        self.ckpt_path = CKPT_PATH + self.model.name + ".ckpt"

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=MAX_LEARNING_RATE)
        self.loss = None

        self.num_epochs = NUM_EPOCHS
        self.lr_step = LR_STEP
        self.max_lr = MAX_LEARNING_RATE
        self.min_lr = MIN_LEARNING_RATE
        self.lr_decline_rate = LR_DECLINE_RATE
        self.lr = MAX_LEARNING_RATE

        self.total = 0
        self.correct = 0
        self.epoch_loss = 0
        self.epoch_losses = None
        self.min_epoch_loss = INF

    def train(self, train_loader):
        for epoch_id in trange(1, self.num_epochs + 1):
            self.run_epoch(loader=train_loader, is_training=True)
            self.save_checkpoint()
            self.update_lr(epoch_id)
        print("checkpoint is saved at " + self.ckpt_path)

    def test(self, test_loader):
        self.run_epoch(loader=test_loader, is_training=False)

    def run_epoch(self, loader, is_training):
        self.model.train() if is_training else self.model.eval()
        self.refresh_eval_and_loss()
        for _, (images, labels) in tqdm(enumerate(loader)):
            if images.size(0) != BATCH_SIZE: continue
            self.run_batch(images, labels)
        self.epoch_loss /= len(loader)
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

    def refresh_eval_and_loss(self):
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

    def save_checkpoint(self):
        if self.min_epoch_loss > self.epoch_loss:
            self.min_epoch_loss = self.epoch_loss
            torch.save(self.model.state_dict(), self.ckpt_path)

    def update_lr(self, epoch_id):
        if epoch_id % self.lr_step == 0: self.max_lr *= self.lr_decline_rate
        self.lr = self.max_lr - (self.max_lr - self.min_lr) * (epoch_id % self.lr_step) / self.lr_step
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
