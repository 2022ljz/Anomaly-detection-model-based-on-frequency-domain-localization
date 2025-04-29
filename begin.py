import time
from utils.utils import *
from model.MyModel import MyModel
from data_factory.data_loader import get_loader_segment
import warnings
warnings.filterwarnings("ignore")


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping():
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0,solver = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.solver = solver

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta :
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'l_num': self.solver.l_num,  # 添加Solver的l_num
            'm_num': self.solver.m_num,  # 添加Solver的m_num
            'h_num': self.solver.h_num,  # 添加Solver的h_num
        }
        torch.save(checkpoint, os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.last_index = None
        self.l_num = torch.tensor(1.0, requires_grad=False)
        self.m_num = torch.tensor(1.0, requires_grad=False)
        self.h_num = torch.tensor(1.0, requires_grad=False)
        self.consecutive_counts = [0, 0, 0] #对应三个位置的连续计数
        self.early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset,solver=self)

    #被动关注权重衰减计数器
    def update_abc(self):
        # 定义衰减规则（可调整）
        decay_factor = 1-100*self.lr  # 学习率为0.0001时 衰减率为0.99
        min_coefficient = 0.001 # 最小系数阈值

        for idx, count in enumerate(self.consecutive_counts):
            if  count >=5:
                # 应用衰减并重置计数器
                if idx == 0:
                    self.b = max(self.b * decay_factor, min_coefficient)
                    self.c = max(self.c * decay_factor, min_coefficient)
                elif idx == 1:
                    self.a = max(self.a * decay_factor, min_coefficient)
                    self.c = max(self.c * decay_factor, min_coefficient)
                else:
                    self.a = max(self.a * decay_factor, min_coefficient)
                    self.b = max(self.b * decay_factor, min_coefficient)
                self.consecutive_counts[idx] = 0

    def build_model(self):
        self.model = MyModel(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, lm, lh, hm = self.model(input)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss).item())
        return np.average(loss_1)

    #被动关注模块实现
    def train(self):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = self.early_stopping
        train_steps = len(self.train_loader)
        self.a = torch.tensor(1.0, requires_grad=False)
        self.b = torch.tensor(1.0, requires_grad=False)
        self.c = torch.tensor(1.0, requires_grad=False)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            epoch_time = time.time()
            self.model.train()
            self.a = 1.0
            self.b = 1.0
            self.c = 1.0
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output,lm,lh,hm = self.model(input)

                lm_loss = 0.0
                lh_loss = 0.0
                hm_loss = 0.0

                for u in range(len(lm)):
                    lm_loss += self.criterion(lm[u],input)
                    lh_loss += self.criterion(lh[u], input)
                    hm_loss += self.criterion(hm[u], input)

                lm_loss = lm_loss / len(lm)
                lh_loss = lh_loss / len(lh)
                hm_loss = hm_loss / len(hm)

                losses = [self.a*lm_loss,self.b*lh_loss,self.c*hm_loss]
                fre_loss = min(losses) #修改点

                #修改点
                fre_index = losses.index(fre_loss)
                if fre_index == 0:
                    self.h_num += 1
                    h_num = 3
                    l_num = 1
                    m_num = 1
                elif fre_index == 1:
                    self.m_num += 1
                    h_num = 1
                    l_num = 1
                    m_num = 3
                else:
                    self.l_num += 1
                    h_num = 1
                    l_num = 3
                    m_num = 1

                #修改点
                # 进行权重调整
                if fre_index == self.last_index:
                    self.consecutive_counts[fre_index] += 1
                else:
                    self.consecutive_counts = [0, 0, 0]
                    self.consecutive_counts[fre_index] = 1
                    self.last_index = fre_index
                self.update_abc()

                print(f'train_low:{self.l_num},high:{self.h_num},mid:{self.m_num}     lm_loss:{lm_loss},lh_loss:{lh_loss},hm_loss:{hm_loss}')

                k1 = (l_num) / (h_num + m_num + l_num) #hm
                k2 = (m_num) / (h_num + m_num + l_num) #hl
                k3 = (h_num) / (h_num + m_num + l_num) #ml
                #print(k1,k2,k3)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss + k1*hm_loss + k2*lh_loss + k3*lm_loss).item())
                loss1 = rec_loss +k1*hm_loss + k2*lh_loss + k3*lm_loss


                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss1.backward()
                self.optimizer.step()

            self.last_index = None
            self.consecutive_counts = [0, 0, 0]

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1,self.model,path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)


    def test(self):
        checkpoint = torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.l_num = checkpoint['l_num']
        self.m_num = checkpoint['m_num']
        self.h_num = checkpoint['h_num']

        self.model.eval()
        num_list = [self.l_num,self.m_num,self.h_num]
        num_list = sorted(num_list)
        self.adjust = (num_list[2] + num_list[1] - 2*num_list[0])//2 + 1

        print("======================TEST MODE======================")
        print(f'test_low:{self.l_num},high:{self.h_num},mid:{self.m_num}')

        criterion = nn.MSELoss(reduction='none')

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, lm, lh, hm = self.model(input)

            lm_loss1 = 0.0
            lh_loss1 = 0.0
            hm_loss1 = 0.0

            for u in range(len(lm)):
                lm_loss1 += torch.einsum('bld,bsd->bls', input, lm[u])
                lh_loss1 += torch.einsum('bld,bsd->bls', input, lh[u])
                hm_loss1 += torch.einsum('bld,bsd->bls', input, hm[u])

            #构建异常强化矩阵
            lm_loss1 = lm_loss1 / len(lm)* self.adjust  * (self.l_num + self.m_num) * (1 / self.h_num)
            lh_loss1 = lh_loss1 / len(lh)* self.adjust  * (self.l_num + self.h_num) * (1 / self.m_num)
            hm_loss1 = hm_loss1 / len(hm)* self.adjust  * (self.h_num + self.m_num) * (1 / self.l_num)
            ##用于消融实验
            # lm_loss1 = lm_loss1 / len(lm)
            # lh_loss1 = lh_loss1 / len(lh)
            # hm_loss1 = hm_loss1 / len(hm)
            # B,L,_ = input.shape
            # lm_loss1 = self.adjust  * (self.l_num + self.m_num) * (1 / self.h_num)*torch.ones(B,L,L).to(self.device)
            # lh_loss1 = self.adjust  * (self.l_num + self.h_num) * (1 / self.m_num)*torch.ones(B,L,L).to(self.device)
            # hm_loss1 = self.adjust  * (self.h_num + self.m_num) * (1 / self.l_num)*torch.ones(B,L,L).to(self.device)
            loss2 = lm_loss1 + lh_loss1 + hm_loss1
            loss = torch.mean(criterion(input, output), dim=-1)

            metric = torch.softmax(torch.mean(-loss2, dim=-1), dim=-1)
            cri = metric * loss
            # cri = loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, lm, lh, hm = self.model(input)

            lm_loss1 = 0.0
            lh_loss1 = 0.0
            hm_loss1 = 0.0

            for u in range(len(lm)):
                lm_loss1 += torch.einsum('bld,bsd->bls', input, lm[u])
                lh_loss1 += torch.einsum('bld,bsd->bls', input, lh[u])
                hm_loss1 += torch.einsum('bld,bsd->bls', input, hm[u])

            lm_loss1 = lm_loss1 / len(lm)* self.adjust  * (self.l_num + self.m_num) * (1 / self.h_num)
            lh_loss1 = lh_loss1 / len(lh)* self.adjust  * (self.l_num + self.h_num) * (1 / self.m_num)
            hm_loss1 = hm_loss1 / len(hm)* self.adjust  * (self.h_num + self.m_num) * (1 / self.l_num)
            ##用于消融实验
            # lm_loss1 = lm_loss1 / len(lm)
            # lh_loss1 = lh_loss1 / len(lh)
            # hm_loss1 = hm_loss1 / len(hm)
            # B,L,_ = input.shape
            # lm_loss1 = self.adjust  * (self.l_num + self.m_num) * (1 / self.h_num)*torch.ones(B,L,L).to(self.device)
            # lh_loss1 = self.adjust  * (self.l_num + self.h_num) * (1 / self.m_num)*torch.ones(B,L,L).to(self.device)
            # hm_loss1 = self.adjust  * (self.h_num + self.m_num) * (1 / self.l_num)*torch.ones(B,L,L).to(self.device)
            loss2 = lm_loss1 + lh_loss1 + hm_loss1

            loss = torch.mean(criterion(input, output), dim=-1)

            metric = torch.softmax(torch.mean(-loss2, dim=-1), dim=-1)
            cri = metric * loss
            # cri = loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, lm, lh, hm = self.model(input)

            lm_loss1 = 0.0
            lh_loss1 = 0.0
            hm_loss1 = 0.0

            for u in range(len(lm)):
                lm_loss1 += torch.einsum('bld,bsd->bls', input, lm[u])
                lh_loss1 += torch.einsum('bld,bsd->bls', input, lh[u])
                hm_loss1 += torch.einsum('bld,bsd->bls', input, hm[u])

            lm_loss1 = lm_loss1 / len(lm)* self.adjust  * (self.l_num + self.m_num) * (1 / self.h_num)
            lh_loss1 = lh_loss1 / len(lh)* self.adjust  * (self.l_num + self.h_num) * (1 / self.m_num)
            hm_loss1 = hm_loss1 / len(hm)* self.adjust  * (self.h_num + self.m_num) * (1 / self.l_num)
            ##用于消融实验
            # lm_loss1 = lm_loss1 / len(lm)
            # lh_loss1 = lh_loss1 / len(lh)
            # hm_loss1 = hm_loss1 / len(hm)
            # B,L,_ = input.shape
            # lm_loss1 = self.adjust  * (self.l_num + self.m_num) * (1 / self.h_num)*torch.ones(B,L,L).to(self.device)
            # lh_loss1 = self.adjust  * (self.l_num + self.h_num) * (1 / self.m_num)*torch.ones(B,L,L).to(self.device)
            # hm_loss1 = self.adjust  * (self.h_num + self.m_num) * (1 / self.l_num)*torch.ones(B,L,L).to(self.device)
            loss2 = lm_loss1 + lh_loss1 + hm_loss1

            loss = torch.mean(criterion(input, output), dim=-1)

            metric = torch.softmax(torch.mean(-loss2, dim=-1), dim=-1)
            cri = metric * loss
            # cri = loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: 沿用一般的评价方法
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        return accuracy, precision, recall, f_score
