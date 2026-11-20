import copy
import random
import time
from flcore.clients.clientfml import clientFML
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from flcore.trainmodel.models import BaseHeadSplit


class FML(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        if args.save_folder_name == 'temp' or args.save_folder_name.startswith('temp_') or 'temp' not in args.save_folder_name:
            if hasattr(args, 'global_model'):
                global_model = BaseHeadSplit(args, is_global=True).to(args.device)
            else:
                global_model = BaseHeadSplit(args, 0).to(args.device)   
            save_item(global_model, self.role, 'global_model', self.save_folder_name)
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFML)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            self.run_clients(self.selected_clients)


            self.receive_ids()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        
        
    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        states = [load_item(self.clients[cid].role, 'global_model', self.clients[cid].save_folder_name).state_dict()
                  for cid in self.uploaded_ids]
        # average every float tensor incl. BatchNorm running stats: averaging parameters() only left the server's BN
        # statistics at their init values, which broke eval-mode use of shared models with BN (ResNet as meme model)
        global_model.load_state_dict({k: (sum(st[k].float() for st in states) / len(states)).to(v.dtype)
                                      if v.is_floating_point() else states[0][k].clone()
                                      for k, v in global_model.state_dict().items()})

        save_item(global_model, self.role, 'global_model', self.save_folder_name)

    def evaluate(self, *args, **kwargs):
        out = super().evaluate(*args, **kwargs)
        d = [c.diag_fml for c in self.clients if getattr(c, "diag_fml", None)]
        if d:
            n = sum(x[2] for x in d)
            print("FMLDiag " + " ".join(f"{k}={sum(x[0][k] for x in d) / n:.4f}" for k in d[0][0])
                  + " | NLL " + " ".join(f"{k}={sum(x[1][k] for x in d) / n:.4f}" for k in d[0][1]))
        return out
