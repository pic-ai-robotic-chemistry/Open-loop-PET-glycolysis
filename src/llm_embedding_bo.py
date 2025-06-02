from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from bayes_opt.util import ensure_rng
import numpy as np
from data import get_one_embedding
import os
import pandas as pd
import openai
from utils import get_openai_key, AcquisitionFunction
import warnings
import json


class LLMEmbeddingBO:

    def __init__(self,
                 embedding_length=1536,
                 random_embedding=20,
                 random_state=None,
                 acquisition='ucb',
                 lazy=False):

        self.embedding_length = embedding_length
        self._random_state = ensure_rng(random_state)
        self.acquisition = acquisition

        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state
        )

        if lazy:
            self.client = None
        else:
            self.client = openai.Client(api_key=get_openai_key())

        self._avail_acid_table = {}
        self._avail_base_table = {}

        self._acid_table = {}
        self._base_table = {}

        self.load_embedding()

        avail_acid_info = pd.read_excel('data/acid.xlsx')['name'].tolist()
        avail_base_info = pd.read_excel('data/base.xlsx')['name'].tolist()

        # load and save cached data
        if os.path.exists('data/all_possible_acid.npy') and os.path.exists(
                'data/all_possible_base.npy'):
            self._all_possible_acid = np.load('data/all_possible_acid.npy')
            self._all_possible_base = np.load('data/all_possible_base.npy')
        else:
            self._all_possible_acid = []
            self._all_possible_base = []

            for acid in avail_acid_info:
                if acid not in self._acid_table.keys():
                    embedding = get_one_embedding(acid, self.client, type='acid')
                    self._all_possible_acid.append(embedding)
                else:
                    embedding = self._acid_table[acid]
                    self._all_possible_acid.append(embedding)
            for base in avail_base_info:
                if base not in self._base_table.keys():
                    embedding = get_one_embedding(base, self.client, type='base')
                    self._all_possible_base.append(embedding)
                else:
                    embedding = self._base_table[base]
                    self._all_possible_base.append(embedding)

            np.save('data/all_possible_acid.npy', self._all_possible_acid)
            np.save('data/all_possible_base.npy', self._all_possible_base)

        for i, acid in enumerate(avail_acid_info):
            self._avail_acid_table[acid] = self._all_possible_acid[i]
        for i, base in enumerate(avail_base_info):
            self._avail_base_table[base] = self._all_possible_base[i]

        self._params = []
        self._targets = []

        if random_embedding is None:
            self.random_embedding_matrix = np.eye(embedding_length * 2)
        else:
            if os.path.exists('data/random_embedding_{}.npy'.format(random_embedding)):
                self.random_embedding_matrix = np.load('data/random_embedding_{}.npy'.format(random_embedding))
            else:
                self.random_embedding_matrix = np.random.randn(random_embedding, embedding_length * 2)
                np.save('data/random_embedding_{}.npy'.format(random_embedding), self.random_embedding_matrix)

        self._evaluated_pair = []
        self.init_register()
        self.save_embedding()

    def save_embedding(self):
        np.save('data/all_possible_acid.npy', self._all_possible_acid)
        np.save('data/all_possible_base.npy', self._all_possible_base)

    def random_embedding(self, x):
        if len(x.shape) == 1:
            x_vector = x.reshape(1, -1)
            return (x_vector @ self.random_embedding_matrix.T)[0]
        else:
            return x @ self.random_embedding_matrix.T

    def restore_random_embedding(self, x):
        if len(x.shape) == 1:
            x_vector = x.reshape(1, -1)
            return (x_vector @ np.linalg.pinv(self.random_embedding_matrix.T))[0]
        else:
            return x @ np.linalg.pinv(self.random_embedding_matrix.T)

    def register(self, param, acid, base, target):
        """
        Register a new observed point into the optimizer.
        :param: input space vector
        :target: target value
        """
        self._params.append(param)
        self._targets.append(target)
        self._evaluated_pair.append((acid, base))

    def register_by_name(self, acid, base, target):
        if acid not in self._acid_table.keys():
            warnings.warn(f"Acid {acid} not found in the embedding table.")
            embedding = get_one_embedding(acid, self.client, type='acid')
            self._acid_table[acid] = embedding
        if base not in self._base_table.keys():
            warnings.warn(f"Base {base} not found in the embedding table.")
            embedding = get_one_embedding(base, self.client, type='base')
            self._base_table[base] = embedding
        param = np.hstack((self._acid_table[acid], self._base_table[base]))
        self.register(param, acid, base, target)

    def load_embedding(self):
        # restore cached data
        data_dir = 'data/embeddings'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        file_names = os.listdir(data_dir)

        for file_name in file_names:
            if not file_name.endswith('.json'):
                continue

            with open(os.path.join(data_dir, file_name), 'r') as f:
                json_data = json.load(f)
                json_data['embedding'] = np.array(json_data['embedding'])

                if json_data['type'] == 'acid':
                    if json_data['molecule'] in self._acid_table:
                        warnings.warn(f"Acid {json_data['molecule']} already exists in the embedding table.")
                    else:
                        self._acid_table[json_data['molecule']] = json_data['embedding']
                else:
                    if json_data['molecule'] in self._base_table:
                        warnings.warn(f"Base {json_data['molecule']} already exists in the embedding table.")
                    else:
                        self._base_table[json_data['molecule']] = json_data['embedding']

    def init_register(self, file=None):

        if file is None:

            rate = pd.read_excel('data/init_experiments.xlsx')

            acid_array = []
            base_array = []

            rate_array_temp = []
            rate_array = []

            for i in range(len(rate)):
                row = rate.iloc[i]
                if not (pd.isna(row['Acid']) and pd.isna(row['Base'])):
                    rate_array.append(np.average(rate_array_temp))
                    rate_array_temp = []
                    if pd.isna(row['Acid']):
                        acid_array.append(row['Base'])
                        base_array.append(row['Base'])
                    elif pd.isna(row['Base']):
                        acid_array.append(row['Acid'])
                        base_array.append(row['Acid'])
                    else:
                        acid_array.append(row['Acid'])
                        base_array.append(row['Base'])

                rate_array_temp.append(row['Yield'])
            rate_array.append(np.average(rate_array_temp))
            del (rate_array[0])

            for i in range(len(acid_array)):
                acid = acid_array[i]
                base = base_array[i]
                if acid in self._acid_table:
                    acid_embedding = self._acid_table[acid]
                elif acid in self._base_table:
                    acid_embedding = self._base_table[acid]
                else:
                    acid_embedding = get_one_embedding(acid, self.client, type='acid')
                    self._acid_table[acid] = acid_embedding
                    self._all_possible_acid = np.vstack((self._all_possible_acid, acid_embedding))
                if base in self._base_table:
                    base_embedding = self._base_table[base]
                elif base in self._acid_table:
                    base_embedding = self._acid_table[base]
                else:
                    base_embedding = get_one_embedding(base, self.client, type='base')
                    self._base_table[base] = base_embedding
                    self._all_possible_base = np.vstack((self._all_possible_base, base_embedding))
                param = np.hstack((acid_embedding, base_embedding))
                target = rate_array[i]
                self.register(param, acid, base, target)

    def fit_gp(self):
        target = np.array(self._targets)
        params = []
        for param in self._params:
            params.append(self.random_embedding(param))
        params = np.array(params)
        self._gp.fit(params, target)

    def suggest(self, batch_size=1):
        acquisition_func = AcquisitionFunction(self.acquisition)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fit_gp()

        y_max = np.max(self._targets)

        best_acq_value = [-np.inf for _ in range(batch_size)]
        best_acid = [None for _ in range(batch_size)]
        best_base = [None for _ in range(batch_size)]

        # Finding the best point to evaluate next by finding the argmax of the acquisition function.
        for acid in self._avail_acid_table.keys():
            for base in self._avail_base_table.keys():
                if (acid, base) in self._evaluated_pair:
                    continue

                acid_embedding = self._avail_acid_table[acid]
                base_embedding = self._avail_base_table[base]
                embedding = np.hstack((acid_embedding, base_embedding))
                random_embedding = self.random_embedding(embedding)
                acq_value = acquisition_func.utility(random_embedding, self._gp, y_max)

                for i in range(batch_size):
                    if acq_value > best_acq_value[i]:
                        best_acq_value[i] = acq_value
                        best_acid[i] = acid
                        best_base[i] = base
                        break

        return best_acid, best_base


if __name__ == '__main__':
    bo = LLMEmbeddingBO()
    print(bo.suggest(batch_size=6))
