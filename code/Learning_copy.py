from generation import Logic_Model_Generator
import numpy as np
import itertools
import scipy.stats as stats
from tqdm import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import itertools

def generate_data(num_sample:int=10, time_horizon:float=0.5):
    gen = Logic_Model_Generator(mu=40,sigma=20,alpha=np.array([7,2,1]))
    data = gen.generate_data(num_sample=num_sample, time_horizon=time_horizon)
    return data

class LSTM_Encoding_Action(nn.Module):

    '''
    input: [batch_size, num_predicate, seq_length]
    Parameters:
        input_size:
        hidden_size:
        output_size:
        num_layers:
    '''

    def __init__(self, input_size, hidden_size, output_size, batch_size, device, num_layers:int = 1) -> None:
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # one-directional LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x:torch.tensor):
        batch_size, num_predicate, seq_length = x.shape
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        x, _  = self.lstm(x, (h_0, c_0)) #NOTE: x:(batch_size, seq_length, num_directions * hidden_size)
        x = self.linear(x)
        x = x.view(batch_size, num_predicate, -1)
        return x

class LSTM_Encoding_History(nn.Module):

    '''
    NOTE:Returns a categorical distribution

    Parameters:
        input_size:
        hidden_size:
        output_size:
        num_layers
    '''

    def __init__(self, input_size, hidden_size, output_size, batch_size, device, num_layers: int = 1) -> None:
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # one-directional LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x:torch.tensor, action_embedding: torch.tensor):
        '''
        Parameters:
            x: mental history
            action_embedding: encoding of action history. This should be the output of LSTM_Encoding_Action
        '''
        batch_size, num_predicate, seq_length = x.shape
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        x, _ = self.lstm(x,(h_0,c_0))                              #NOTE: x:(batch_size, seq_length, num_directions * hidden_size)
        x = torch.concat(tensors=[x, action_embedding], dim=1)     #NOTE: concatenate the action info and the mental info
        batch_size, num_predicate, hidden_size = x.shape           #NOTE: num_predicate is changed
        x = self.linear(x)                                         #NOTE: x:(batch_size, seq_length, num_directions * output_size)
        x = x.view(batch_size, num_predicate, -1)
        #TODO: return a vector with dimension (I+1), (I represents the number of types of mental states)
        #TODO: the i-th (i=0,1,2,...,I) component of the output x represents the probability of the i-th mental type
        x = self.softmax(x)
        return x.view(batch_size,-1,self.output_size).mean(axis=1)

class Bayesian_TLPP:
    
    def __init__(self, time_horizon:float, action_history:dict, hidden_size:tuple, output_size:tuple, batch_size:int, partition_size:float=1.2, device:str='cuda', num_layers:tuple=(1,1),kappa0:int = 1, v0:int = 1, mu0:int=40, sigma0sq:int=400, alpha:np.array=np.array([1,1,1])):
        self.kappa0 = kappa0
        self.v0 = v0
        self.mu0 = mu0
        self.sigma0sq = sigma0sq
        self.alpha = alpha
        self.num_disease = len(alpha)

        self.alpha_A = np.array([1,1])
        self.alpha_I = np.array([1,1])
        self.alpha_R = np.array([1,1])
        self.theta0 = -0.4*np.ones(shape=(3,))
        self.tau0sq = 0.04*np.eye(3)
        self.delta = 0.08*np.eye(3)


        self.time_horizon = time_horizon
        self.partition_size = partition_size            # num of small time intervals
        self.action_history = action_history
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device
        self.num_layers = num_layers                    # num_layers of LSTMs

        ### the following parameters are used to manually define the logic rules
        self.num_predicate = 3                  # num_predicate is same as num_node
        self.num_formula = 3                    # num of prespecified logic rules
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.Time_tolerance = 0.3               
        self.body_predicate_set = []            # the index set of all body predicates
        self.mental_predicate_set = [1]
        self.action_predicate_set = [0,2]
        self.head_predicate_set = [0, 1, 2]     # the index set of all head predicates
        self.decay_rate = 0.3                                 # decay kernel
        self.integral_resolution = 0.05


        #TODO: convert the action_history:dict to a numpy array 'processed_data':np.array to put in the LSTMs
        self.processed_data = self.process_data(action_history=self.action_history).to(device)
        self.INPUT_SIZE_A = self.processed_data.shape[-1]
        #self.INPUT_SIZE_M = int(self.time_horizon / self.partition_size)

        #TODO: construct two LSTMs to encode the past history
        #NOTE: encoding action history
        self.LSTM_Action = LSTM_Encoding_Action(input_size=self.INPUT_SIZE_A,hidden_size=hidden_size[0],output_size=output_size[0],batch_size=batch_size,device=device,num_layers=self.num_layers[0])
        self.LSTM_Action.to(device)
        #NOTE: encoding whole history
        self.LSTM_History = LSTM_Encoding_History(input_size=len(self.mental_predicate_set),hidden_size=hidden_size[1],output_size=output_size[1],batch_size=batch_size ,device=device,num_layers=self.num_layers[1])
        self.LSTM_History.to(device)


        ### the following parameters are used to generate synthetic data
        ### for the learning part, the following is used to claim variables
        ### self.model_parameter = {0:{},1:{},...,6:{}}
        self.model_parameter = {}

        head_predicate_idx = 0
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = -0.5

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0.8

        formula_idx = 1
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0.2

        head_predicate_idx = 1
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = -0.24

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0.3

        formula_idx = 1
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0.7


        head_predicate_idx = 2
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = -0.19

        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0.3

        formula_idx = 1
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0.7

        
        #NOTE: set the content of logic rules
        self.logic_template = self.logic_rule()

    def process_data(self, action_history:dict)->torch.tensor:
        '''
        Parameters:
            action_history: dict
        '''
        #TODO: convert the action sequences into a numpy array
        #NOTE: action_history = {
        #                       3: {...}
        #                       4: {...}
        #                       5: {...}
        #                       6: {...}
        #                       }
        # "..." stands for transition times for predicate 3,4,5,6. Recall that 3,4,5,6 are all action predicates
        result = []
        max_action_transition_time_length = 0       #NOTE: record the length of the transition time
        for sample_id in action_history:            #NOTE: batch
            for action_predicate_idx in self.action_predicate_set:
                #print(sample_id, action_predicate_idx)
                time = action_history[sample_id][action_predicate_idx]['time'][1:]
                tmp = len(time)
                if tmp > max_action_transition_time_length: max_action_transition_time_length = tmp
                result.append(time)
        #print(result)
        #NOTE: shape (batch_size:len(action_history), num_predicate:len(self.action_predicate_set), seq_length:max_action_transition_time_length)
        data = np.zeros(shape=(len(action_history),len(self.action_predicate_set),max_action_transition_time_length))
        #TODO: store the action history in a tensor
        for batch in range(data.shape[0]):
            for row in range(data.shape[1]): 
                data[batch, row, :len(result[(batch+1)*row])] = result[(batch+1)*row]
        return torch.tensor(data).float()

    def logic_rule(self):
        #TODO: the logic rules encode the prior knowledge
        # encode rule information
        '''
        This function encodes the content of logic rules
        logic_template = {0:{},1:{},...,6:{}}
        '''
        logic_template = {}


        '''
        Mental (0)
        '''

        head_predicate_idx = 0
        logic_template[head_predicate_idx] = {} # here 0 is the index of the head predicate; we could have multiple head predicates

        #NOTE: rule content: 1 and 2 and before(1, 0) and before(2,0) \to \neg 0
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1, 2]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1, 1]  # use 1 to indicate True; use -1 to indicate False
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [0]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1, 0], [2, 0]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE, self.BEFORE]

        #NOTE: rule content: 1 and \neg 2 and before(1,0) \to 0 
        formula_idx = 1
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1, 2]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1, 0]  # use 1 to indicate True; use -1 to indicate False
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1, 0]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]


        '''
        Action (1-2)
        '''

        head_predicate_idx = 1
        logic_template[head_predicate_idx] = {}  # here 1 is the index of the head predicate; we could have multiple head predicates

        #NOTE: rule content: 0 and before(0,1) to 1
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 1]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        #NOTE: rule content: 2 and before(2,1) to \neg 1
        formula_idx = 1
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [0]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2, 1]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        head_predicate_idx = 2
        logic_template[head_predicate_idx] = {}  # here 2 is the index of the head predicate; we could have multiple head predicates

        #NOTE: rule content: 0 and before(0, 2) to 2
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 2]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        #NOTE: rule content: 1 and before(1, 2) to 2
        formula_idx = 1
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [0]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1, 2]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        return logic_template

    def intensity(self, cur_time, head_predicate_idx, history):
        feature_formula = []
        weight_formula = []
        effect_formula = []
        #TODO: Check if the head_prediate is a mental predicate
        if head_predicate_idx in self.mental_predicate_set: flag = 0
        else: flag = 1  #NOTE: action
        #print(head_predicate_idx)
        for formula_idx in list(self.logic_template[head_predicate_idx].keys()): # range all the formula for the chosen head_predicate
            weight_formula.append(self.model_parameter[head_predicate_idx][formula_idx]['weight'])
            feature_formula.append(self.get_feature(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                    history=history, template=self.logic_template[head_predicate_idx][formula_idx], flag=flag))
            effect_formula.append(self.get_formula_effect(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                       history=history, template=self.logic_template[head_predicate_idx][formula_idx]))
        intensity = np.exp(np.array(weight_formula))/ np.sum( np.exp(np.array(weight_formula)), axis=0) * np.array(feature_formula) * np.array(effect_formula)

        intensity = self.model_parameter[head_predicate_idx]['base'] + np.sum(intensity)
        intensity = np.exp(intensity)
        #print(head_predicate_idx, intensity)
        return intensity

    def get_feature(self, cur_time, head_predicate_idx, history, template, flag:int):
        #NOTE: flag: 0 or 1, denotes the head_predicate_idx is a mental or an action
        #NOTE: 0 for mental and 1 for action
        #NOTE: since for mental, we need to go through all the history information
        #NOTE: while for action, we only care about the current time information
        transition_time_dic = {}
        feature = 0
        for idx, body_predicate_idx in enumerate(template['body_predicate_idx']):
            transition_time = np.array(history[body_predicate_idx]['time'])
            transition_state = np.array(history[body_predicate_idx]['state'])
            mask = (transition_time <= cur_time) * (transition_state == template['body_predicate_sign'][idx]) # find corresponding history
            transition_time_dic[body_predicate_idx] = transition_time[mask]
        transition_time_dic[head_predicate_idx] = [cur_time]
        ### get weights
        # compute features whenever any item of the transition_item_dic is nonempty
        history_transition_len = [len(i) for i in transition_time_dic.values()]
        if min(history_transition_len) > 0:
            # need to compute feature using logic rules
            time_combination = np.array(list(itertools.product(*transition_time_dic.values()))) # get all possible time combinations
            time_combination_dic = {}
            for i, idx in enumerate(list(transition_time_dic.keys())):
                #TODO: this is where we distinguish mental and action
                time_combination_dic[idx] = time_combination[:, i] if flag == 0 else time_combination[-1, i]
            temporal_kernel = np.ones(len(time_combination))
            for idx, temporal_relation_idx in enumerate(template['temporal_relation_idx']):
                time_difference = time_combination_dic[temporal_relation_idx[0]] - time_combination_dic[temporal_relation_idx[1]]
                if template['temporal_relation_type'][idx] == 'BEFORE':
                    temporal_kernel *= (time_difference < - self.Time_tolerance) * np.exp(-self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'EQUAL':
                    temporal_kernel *= (abs(time_difference) <= self.Time_tolerance) * np.exp(-self.decay_rate *(cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'AFTER':
                    temporal_kernel *= (time_difference > self.Time_tolerance) * np.exp(-self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[1]]))
            feature = np.sum(temporal_kernel)
            #print(head_predicate_idx, feature)
        return feature

    def get_formula_effect(self, cur_time, head_predicate_idx, history, template):
        ## Note this part is very important!! For generator, this should be np.sum(cur_time > head_transition_time) - 1
        ## Since at the transition times, choose the intensity function right before the transition time
        head_transition_time = np.array(history[head_predicate_idx]['time'])
        head_transition_state = np.array(history[head_predicate_idx]['state'])
        if len(head_transition_time) == 0:
            cur_state = 0
            counter_state = 1 - cur_state
        else:
            idx = np.sum(cur_time > head_transition_time) - 1
            cur_state = head_transition_state[idx]
            counter_state = 1 - cur_state

        if counter_state == template['head_predicate_sign']:
            formula_effect = 1              # the formula encourages the head predicate to transit
        else:
            formula_effect = -1
        return formula_effect

    def log_likelihood(self, dataset, sample_ID_batch, T_max, p, mu, sigma):
        '''
        This function calculates the log-likehood given the dataset
        log-likelihood = \sum log(intensity(transition_time)) + int_0^T intensity dt

        Parameters:
            dataset: 
            sample_ID_batch: list
            T_max:
        '''
        log_likelihood = 0
        # iterate over samples
        for sample_ID in sample_ID_batch:
            # iterate over head predicates; each predicate corresponds to one intensity
            data_sample = dataset[sample_ID]
            for head_predicate_idx in self.head_predicate_set:
                #NOTE: compute the summation of log intensities at the transition times
                intensity_log_sum = self.intensity_log_sum(head_predicate_idx, data_sample)
                intensity_log_sum_marker = self.intensity_log_sum_marker(head_predicate_idx,data_sample,p,mu,sigma)
                #NOTE: compute the integration of intensity function over the time horizon
                intensity_integral = self.intensity_integral(head_predicate_idx, data_sample, T_max)

                log_likelihood += (intensity_log_sum + intensity_log_sum_marker - intensity_integral)
        #print(log_likelihood)
        return np.exp(log_likelihood)

    def intensity_log_sum(self, head_predicate_idx, data_sample):
        intensity_transition = []
        for t in data_sample[head_predicate_idx]['time'][1:]:
            #NOTE: compute the intensity at transition times
            cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
            intensity_transition.append(cur_intensity)
        if len(intensity_transition) == 0: # only survival term, no event happens
            log_sum = 0
        else:
            log_sum = np.sum(np.log(np.array(intensity_transition)))
        return log_sum

    def intensity_log_sum_marker(self, head_predicate_idx, data_sample, p, mu, sigma):
        result = 0
        for marker in data_sample[head_predicate_idx]['marker'][1:]:
            #NOTE: range over all the markers
            d = marker[0:self.num_disease]     #TODO: get the disease type
            a = marker[-1]                     #TODO: get age
            result += np.log(stats.multinomial.pmf(d,n=1,p=p) * stats.norm(loc=mu, scale=sigma).pdf(a))
        return result

    def intensity_integral(self, head_predicate_idx, data_sample, T_max):
        start_time = 0
        end_time = T_max
        intensity_grid = []
        for t in np.arange(start_time, end_time, self.integral_resolution):
            #NOTE: evaluate the intensity values at the chosen time points
            cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
            intensity_grid.append(cur_intensity)
        #NOTE: approximately calculate the integral
        integral = np.sum(np.array(intensity_grid) * self.integral_resolution)
        return integral

    def update_posterior_parameters(self, data):
        event_count = 0 #TODO: count the total number of events
        r_record = []
        d_count = np.zeros(shape=(3))      
        for sample_id in data:
            for predicate_idx in data[sample_id]:
                event_count += len(data[sample_id][predicate_idx]['time'][1:])
                for marker in data[sample_id][predicate_idx]['marker'][1:]:
                    d = marker[:3]
                    r = marker[-1]
                    idx = np.argmax(d)
                    d_count[idx] += 1
                    r_record.append(r)
        r_record = np.array(r_record)
        #TODO: update
        alpha = self.alpha + d_count
        kappa = self.kappa0 + event_count
        v = self.v0 + event_count
        mu = (self.kappa0 * self.mu0 + np.sum(r_record)) / (kappa)
        sigmasq = (self.v0 * self.sigma0sq + np.sum( (r_record - r_record.mean())**2 ) + (self.kappa0*event_count)*(r_record.mean()-self.mu0)**2 / kappa) / v
        return alpha, kappa, v, mu, sigmasq

    def imputing(self,sample_ID_batch:list, temperature:float=1.0, device='cuda')->torch.tensor:
        #NOTE: we add a small time shift 1e-4 so that we can include the end time point in 'time_intervals'
        time_intervals = np.arange(0,self.time_horizon+1e-4,step=self.partition_size)
        #print(time_intervals) #NOTE: checkpoint

        #TODO: initialize. Store the complete data
        complete_history = dict([(sample_id, self.action_history[sample_id]) for sample_id in sample_ID_batch])
        for sample_id in complete_history:
            for mental_predicate_idx in self.mental_predicate_set:
                complete_history[sample_id][mental_predicate_idx] = {}
                complete_history[sample_id][mental_predicate_idx]['time'] = [0]
                complete_history[sample_id][mental_predicate_idx]['state'] = [0]
        #print(complete_history) #NOTE: checkpoint
        #TODO: initilize, store mental history, this will be fed into the LSTM
        mental_history = torch.zeros(size=(len(sample_ID_batch),1,len(self.mental_predicate_set))).to(device)

        #TODO: encode the mental history
        h_a = self.LSTM_Action.forward(self.processed_data[sample_ID_batch,:,:]) #h_a (batch_size, num_predicate, output_size=hidden_size_m)
        for i in range(len(time_intervals)-1):
            #TODO: encode the mental history before the i-th time interval (LSTMs) -> categorical distribution -> prob
            # prob (batch_size, 1, num_mental_predicate+1)
            prob:torch.tensor = self.LSTM_History.forward(x=mental_history,action_embedding=h_a)
            # logits (batch_size, 1, num_mental_predicate+1)
            logits:torch.tensor = torch.log(prob)
            #TODO: draw hard samples. post_samples = self.sample_variational_posterior_hard(size = sample_size, prob = prob)
            post_samples:torch.tensor = self.sample_variational_posterior(size=len(sample_ID_batch),logits=logits,temperature=temperature,hard=True)
            #TODO: after sampling the mental transition time, update the history information
            event_time = (time_intervals[i] + time_intervals[i+1])/2
            _, indices = post_samples.max(dim=2)
            indices = indices.detach().cpu().numpy()
            new_mental_information = torch.zeros(size=(len(sample_ID_batch),1,len(self.mental_predicate_set))).to(device)

                    #print(indices[0,:])
            for batch_idx, idx in enumerate(indices[0]):
                if idx == 0: continue   #NOTE: no mental
                #TODO: We have mental
                #TODO: update the mental history
                new_mental_information[batch_idx,0,idx-1] = event_time
                #TODO: update the complete history, which is a dict
                sample_id = sample_ID_batch[batch_idx]
                complete_history[sample_id][1]['time'].append(event_time)
                #print(idx-1)   #NOTE: checkpoint
                if complete_history[sample_id][1]['state'][-1] == 0: complete_history[sample_id][1]['state'].append(1)
                else: complete_history[sample_id][1]['state'].append(0)
            #print(new_mental_information) #NOTE:checkpoint
            #TODO: update mental_history
            mental_history = torch.concat([mental_history,new_mental_information],dim=1)
        #print(complete_history)

        return complete_history

    def sample_variational_posterior(self, size:int, logits: torch.tensor, temperature:float=1.0, hard=False)->torch.tensor:
        #TODO: use gumbel-max trick to explicitly sample from the variational posterior defined by LSTMs, which is a categorical distribution
        '''
        draw explicit samples from variational posterior

        Parameters:
            size: number of samples
            logits: 
            hard: boolean
        '''
    
        result = []
        for i in range(size):
            tmp = self.gumbel_softmax(logits,temperature,hard)
            result.append(tmp)
        #NOTE: return one-hot vectors.
        #print(result)
        result = torch.stack(result,dim=0)
        return result

    def sample_Gumble(self, shape, eps:float=1e-20):
        #TODO: sample from Gumbel(0,1). This is needed when we want to explicitly sample from the variational posterior
        '''
        Sample from Gumbel(0,1) with shape = 'shape'

        Parameters:
            shape: 
            eps: small perturbation to avoid log(0)
            tens_type: 
        '''

        U = torch.rand(shape)
        U = U.cuda()
        return -torch.log(-torch.log(U+eps)+eps)

    def sample_Gumble_softmax(self, logits:torch.tensor, temperature=1.0):
        #TODO: sample from the gumbel-softmax distribution
        '''
        Parameters:
            prob: 
            temperature:
        '''
        y = logits + self.sample_Gumble(logits.shape)
        return F.softmax(y/temperature,dim=-1)

    def gumbel_softmax(self, logits:torch.tensor, temperature:float=1.0, hard=False):
        #TODO: ST-gumbel-softmax
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """

        y = self.sample_Gumble_softmax(logits,temperature)
        if not hard: return y
        shape = y.size()
        _, idx = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1,shape[-1])
        y_hard.scatter_(1, idx.view(-1,1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def imputing_markers(self, data:dict, alpha, kappa, v, mu, sigmasq):
        p_cur = np.random.dirichlet(alpha)
        sigmasq_cur = 1/np.random.gamma(v/2,1/(v*sigmasq/2))
        mu_cur = np.random.normal(loc=mu,scale=np.sqrt(sigmasq/kappa))
        for sample_ID in data:
            for idx in self.mental_predicate_set:
                data[sample_ID][idx]['marker'] = []
        for sample_ID in data:
            for idx in self.mental_predicate_set:
                n = len(data[sample_ID][idx]['time'][1:])
                for j in range(n):
                    d = np.random.multinomial(n=1,pvals=p_cur)
                    r = np.random.normal(loc=mu_cur,scale=np.sqrt(sigmasq_cur))
                    marker = np.hstack([d,np.array(r)])
                    data[sample_ID][idx]['marker'].append(marker)
        return data

    def Gibbs(self, data, initial_guess:list, T_max, num_iters:int=1, inner_iters:int=10):
        batch_size=self.batch_size
        alpha, kappa, v, mu, sigmasq = self.update_posterior_parameters(data)
        result = {'omegaA':[],'omegaI':[],'omegaR':[],'B':[],'p':[],'mu':[],'sigmasq':[]}
        omegaA_cur = initial_guess[0]
        omegaI_cur = initial_guess[1]
        omegaR_cur = initial_guess[2]
        B_cur = initial_guess[3]

        #print(v/2,v*sigmasq/2)

        # set initial guess for the model parameters
        self.model_parameter[0][0]['weight'] = omegaA_cur[0]
        self.model_parameter[0][1]['weight'] = omegaA_cur[1]
        self.model_parameter[1][0]['weight'] = omegaI_cur[0]
        self.model_parameter[1][1]['weight'] = omegaI_cur[1]
        self.model_parameter[2][0]['weight'] = omegaR_cur[0]
        self.model_parameter[2][1]['weight'] = omegaR_cur[1]
        self.model_parameter[0]['base'] = B_cur[0]
        self.model_parameter[1]['base'] = B_cur[1]
        self.model_parameter[2]['base'] = B_cur[2]

        B_cur = initial_guess[3]
        for iter in tqdm(range(num_iters)):
            random_idx = list(np.random.randint(0,len(data),size=batch_size))
            #TODO: sample from the variational distribution
            data = self.imputing(sample_ID_batch=random_idx)
            data = self.imputing_markers(data,alpha,kappa,v,mu,sigmasq)
            
            p_cur = np.random.dirichlet(alpha)
            sigmasq_cur = 1/np.random.gamma(v/2,1/(v*sigmasq/2))
            mu_cur = np.random.normal(loc=mu,scale=np.sqrt(sigmasq/kappa))
            result['p'].append(p_cur)
            result['mu'].append(mu_cur)
            result['sigmasq'].append(sigmasq_cur)

            #TODO: sample omega_A
            omegaA_proposal = np.random.dirichlet(alpha=self.alpha_A+10*omegaA_cur)
            #print(omegaA_cur)
            
            a = (np.log(stats.dirichlet(self.alpha_A).pdf(omegaA_cur)) + self.log_likelihood(dataset=data,sample_ID_batch=random_idx,T_max=T_max,p=p_cur,mu=mu_cur,sigma=np.sqrt(sigmasq_cur)) + np.log(stats.dirichlet.pdf(omegaA_proposal,alpha=self.alpha_A+10*omegaA_cur)))
            
            self.model_parameter[0][0]['weight'] = omegaA_proposal[0]
            self.model_parameter[0][1]['weight'] = omegaA_proposal[1]
            
            b = (np.log(stats.dirichlet(self.alpha_A).pdf(omegaA_proposal)) + self.log_likelihood(dataset=data,sample_ID_batch=random_idx,T_max=T_max,p=p_cur,mu=mu_cur,sigma=np.sqrt(sigmasq_cur)) + np.log(stats.dirichlet.pdf(omegaA_cur,alpha=self.alpha_A+10*omegaA_proposal)))
            a = np.exp(b-a)
            #print(a)
            #print(np.min([1,a]))
            if (np.random.random() < np.min([1,a])): omegaA_cur = omegaA_proposal
            result['omegaA'].append(omegaA_cur)
            #TODO

            self.model_parameter[0][0]['weight'] = omegaA_cur[0]
            self.model_parameter[0][1]['weight'] = omegaA_cur[1]

            #TODO: sample omega_I
            omegaI_proposal = np.random.dirichlet(alpha=self.alpha_I+10*omegaI_cur)
            
            a = (np.log(stats.dirichlet(self.alpha_I).pdf(omegaI_cur)) + self.log_likelihood(dataset=data,sample_ID_batch=random_idx,T_max=T_max,p=p_cur,mu=mu_cur,sigma=np.sqrt(sigmasq_cur)) + np.log(stats.dirichlet.pdf(omegaI_proposal,alpha=self.alpha_I+10*omegaI_cur)))
            
            self.model_parameter[1][0]['weight'] = omegaI_proposal[0]
            self.model_parameter[1][1]['weight'] = omegaI_proposal[1]
            
            b = (np.log(stats.dirichlet(self.alpha_I).pdf(omegaI_proposal)) + self.log_likelihood(dataset=data,sample_ID_batch=random_idx,T_max=T_max,p=p_cur,mu=mu_cur,sigma=np.sqrt(sigmasq_cur)) + np.log(stats.dirichlet.pdf(omegaI_cur,alpha=self.alpha_I+10*omegaI_proposal)))
            a = np.exp(b-a)
            #print(a)
            #print(np.min([1,a]))
            if (np.random.random() < np.min([1,a])): omegaI_cur = omegaI_proposal
            result['omegaI'].append(omegaI_cur)
            #TODO:update model parameter

            self.model_parameter[1][0]['weight'] = omegaI_cur[0]
            self.model_parameter[1][1]['weight'] = omegaI_cur[1]

            #TODO: sample omega_R
            omegaR_proposal = np.random.dirichlet(alpha=self.alpha_R+10*omegaR_cur)

            a = (np.log(stats.dirichlet(self.alpha_R).pdf(omegaR_cur)) + self.log_likelihood(dataset=data,sample_ID_batch=random_idx,T_max=T_max,p=p_cur,mu=mu_cur,sigma=np.sqrt(sigmasq_cur)) + np.log(stats.dirichlet.pdf(omegaR_proposal,alpha=self.alpha_R+10*omegaR_cur)))
            
            self.model_parameter[2][0]['weight'] = omegaR_proposal[0]
            self.model_parameter[2][1]['weight'] = omegaR_proposal[1]
            
            b = (np.log(stats.dirichlet(self.alpha_R).pdf(omegaR_proposal)) + self.log_likelihood(dataset=data,sample_ID_batch=random_idx,T_max=T_max,p=p_cur,mu=mu_cur,sigma=np.sqrt(sigmasq_cur)) + np.log(stats.dirichlet.pdf(omegaR_cur,alpha=self.alpha_R+10*omegaR_proposal)))
            a = np.exp(b-a)
            #print(a)
            #print(np.min([1,a]))
            if (np.random.random() < np.min([1,a])): omegaR_cur = omegaR_proposal
            result['omegaR'].append(omegaR_cur)
            #TODO

            self.model_parameter[2][0]['weight'] = omegaR_cur[0]
            self.model_parameter[2][1]['weight'] = omegaR_cur[1]

            #TODO: sample B
            B_proposal = np.random.multivariate_normal(mean=B_cur, cov=self.delta)
            #print(B_proposal)
            
            a = (np.log(stats.multivariate_normal.pdf(B_cur,mean=self.theta0,cov=self.tau0sq)) + self.log_likelihood(dataset=data,sample_ID_batch=random_idx,T_max=T_max,p=p_cur,mu=mu_cur,sigma=np.sqrt(sigmasq_cur)) + np.log(stats.multivariate_normal.pdf(B_proposal,mean=B_proposal,cov=self.delta)))
            
            self.model_parameter[0]['base'] = B_proposal[0]
            self.model_parameter[1]['base'] = B_proposal[1]
            self.model_parameter[2]['base'] = B_proposal[2]
            
            b = (np.log(stats.multivariate_normal.pdf(B_proposal,mean=self.theta0,cov=self.tau0sq)) + self.log_likelihood(dataset=data,sample_ID_batch=random_idx,T_max=T_max,p=p_cur,mu=mu_cur,sigma=np.sqrt(sigmasq_cur)) + np.log(stats.multivariate_normal.pdf(B_cur,mean=B_cur,cov=self.delta)))
            a = np.exp(b-a)
            #print(a)
            #print(np.min([1,a]))
            if (np.random.random() < np.min([1,a])): B_cur = B_proposal
            result['B'].append(B_cur)
            #TODO

            self.model_parameter[0]['base'] = B_cur[0]
            self.model_parameter[1]['base'] = B_cur[1]
            self.model_parameter[2]['base'] = B_cur[2]

            print('iter >>> {}/{}; posterior sample >>> {}'.format(iter+1,num_iters,(omegaA_cur,omegaI_cur,omegaR_cur,B_cur,p_cur,mu_cur,sigmasq_cur)))
        return result

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    np.random.seed(2022)
    def generate_incomplete_data(num_sample:int=10, time_horizon:float=0.5):
        gen = Logic_Model_Generator(mu=40,sigma=20,alpha=np.array([7,2,1]))
        data = gen.generate_data(num_sample=num_sample, time_horizon=time_horizon)
        action_history = {}
        for i in range(num_sample):
            action_history_ = dict([(key, data[i][key]) for key in [0,2]])
            action_history[i] = action_history_
        #NOTE: info
        print('[INFO] data has been generated!!!')
        return action_history
    
    #data = generate_data(num_sample=16,time_horizon=20)
    #print('[INFO] data is generated!!!')
    #model.likelihood(dataset=data,sample_ID_batch=[0,1,2,3,4,5],T_max=5,p=np.array([0.2,0.7,0.1]),mu=35.5,sigma=24.3)
    #model.update_posterior_parameters(data)
    #np.save('data.npy',data)

    data = generate_incomplete_data(num_sample=512,time_horizon=20)

    model = Bayesian_TLPP(time_horizon=20,action_history=data,hidden_size=(15,20),output_size=(20,2),batch_size=1,num_layers=(1,1))
    #complete_data = model.imputing(sample_ID_batch=np.arange(0,10,1))
    #print(complete_data[0])
    res = model.Gibbs(data,initial_guess=[np.array([0.7,0.3]),np.array([0.2,0.8]),np.array([0.3,0.7]),np.array([-0.4,-0.2,-0.3])], T_max=20, num_iters=5000, inner_iters=300)
    np.save('latent_process_data.npy',data)
    np.save('latent_process_result.npy',res)