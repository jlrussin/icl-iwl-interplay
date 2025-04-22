import os
import json
import math
import torch
import random
import numpy as np

from prompts import *
from itertools import permutations, product
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import LlamaTokenizer

class GridTask:
    def __init__(self, save_data_to, n_train, n_val, n_test,
                 n_positions, n_symbols, pt_rot_cond, pt_cond, save_vocab,
                 p_flip_xy, strict_train, strict_test, generate_all, 
                 real_words, preface_id, template_id, sep_id, test_preface_id,
                 lm_task):
        self.save_data_to = save_data_to # path to directory to save all data
        self.n_train = n_train # number of episodes to use for pretraining
        self.n_val = n_val # number of val episodes to use for pretesting
        self.n_test = n_test # number of episodes to use for finetuning
        self.n_positions = n_positions # number of positions per item
        self.n_symbols = n_symbols # number of symbols in alphabet
        self.pt_cond = pt_cond # condition in pretrain phase
        self.save_vocab = save_vocab # path to save vocab to
        self.p_flip_xy = p_flip_xy # probability of flipping (x,y)
        self.strict_train = strict_train # no repeats in training set
        self.strict_test = strict_test # no training samples in test set
        self.generate_all = generate_all # gen all and sample w/o replacement
        self.real_words = real_words # whether to use real color/animal words
        self.preface_id = preface_id # which preface to use (prompts.py)
        self.template_id = template_id # which template to use (prompts.py)
        self.sep_id = sep_id # which sep token to use (prompts.py)
        self.test_preface_id = test_preface_id # which test preface to use
        self.lm_task = lm_task # use masked or causal language modeling task
        self.preface = prefaces[preface_id]
        self.template = templates[template_id]
        self.sep = seps[sep_id]
        self.test_preface = test_prefaces[test_preface_id]

        if pt_rot_cond == 'unrotated':
            self.pt_rotated = False # unrotated in pretrain phase
        elif pt_rot_cond == 'rotated':
            self.pt_rotated = True # rotated in pretrain phase
        if save_vocab is not None:
            self.vocab = {'<pad>': 0}
            self.vocab_idx = 1 # current index to assign to next token
        if strict_train or strict_test:
            self.hash_values = [] # store hash values to ensure no collisions
        if self.real_words:
            assert self.n_positions == 1
            assert self.n_symbols <= len(color_words)
            assert self.n_symbols <= len(animal_words)
            self.colors = [color_words[i] for i in range(self.n_symbols)]
            self.animals = [animal_words[i] for i in range(self.n_symbols)]
        else:
            self.colors = [f'color_{idx}' for idx in range(self.n_symbols)]
            self.animals = [f'animal_{idx}' for idx in range(self.n_symbols)]
        if generate_all:
            # Check we have enough possible samples
            n = self.n_symbols**self.n_positions
            p = math.factorial(n)/math.factorial(n - 5)
            total_possible = p**2
            total_needed = self.n_train + self.n_val + 8*self.n_test
            assert total_needed <= total_possible
            # Generate all possible episodes, to be retrieved in order
            self.generate_all_episodes()
            self.current_episode_id = 0

    def make_datasets(self):
        # Generate episodes for pretraining
        train_episodes = self.generate(self.n_train, self.pt_cond, 
                                       rotated=self.pt_rotated, 
                                       strict=self.strict_train)
        val_episodes = self.generate(self.n_val, self.pt_cond,
                                     rotated=self.pt_rotated, 
                                     strict=self.strict_test)
        # Generate episodes for finetuning
        unrotated_test = {}
        rotated_test = {}
        for ft_cond in ['aligned', 'misaligned', 'blocked', 'interleaved']:
            # Unrotated
            unrotated_episodes = self.generate(self.n_test, ft_cond,
                                               rotated=False, 
                                               strict=self.strict_test)
            unrotated_test[ft_cond] = unrotated_episodes
            
            # Rotated
            rotated_episodes = self.generate(self.n_test, ft_cond,
                                             rotated=True, strict=False)
            rotated_test[ft_cond] = rotated_episodes

        # Make datasets for pretraining
        pretrain_path = os.path.join(self.save_data_to, 'pretrain') 
        self.make_pretrain(train_episodes, pretrain_path)
        pretest_path = os.path.join(self.save_data_to, 'pretest')
        self.make_pretrain(val_episodes, pretest_path)

        # Make datasets for finetuning
        for ft_cond in ['aligned', 'misaligned', 'blocked', 'interleaved']:
            # Unrotated
            cond_dir = f'finetune_unrotated_{ft_cond}/'
            unrotated_path = os.path.join(self.save_data_to, cond_dir)
            if not os.path.isdir(unrotated_path):
                os.mkdir(unrotated_path)
            self.make_finetune(unrotated_test[ft_cond], unrotated_path)

            # Rotated
            cond_dir = f'finetune_rotated_{ft_cond}/'
            rotated_path = os.path.join(self.save_data_to, cond_dir)
            if not os.path.isdir(rotated_path):
                os.mkdir(rotated_path)
            self.make_finetune(rotated_test[ft_cond], rotated_path)

        # Vocabulary
        if self.save_vocab is not None:
            print(f"Saving vocab to {self.save_vocab}", flush=True)
            with open(self.save_vocab, 'w') as f:
                json.dump(self.vocab, f)

    def generate(self, n_episodes, condition, rotated=False, strict=False):
        episodes = []

        # Generated episodes
        for episode_i in range(n_episodes):
            episode = self.generate_episode(condition, rotated=rotated, 
                                            strict=strict)
            episodes.append(episode)
        
        return episodes
    
    def generate_all_episodes(self):
        colors = self.colors
        animals = self.animals
        self.all_episodes = []
        for c, a in product(permutations(colors), permutations(animals)):
            self.all_episodes.append((list(c),list(a)))
        random.shuffle(self.all_episodes)

    def generate_episode(self, condition, rotated=False, strict=False):
        # Colors and animals
        n = 5 # number of colors = number of animals

        def sample_colored_animals():
            if self.generate_all:
                colors, animals = self.all_episodes[self.current_episode_id]
                self.current_episode_id += 1
            else:
                colors = []
                animals = []
                while len(colors) < n:
                    idxs = random.sample(range(self.n_symbols), 
                                         self.n_positions)
                    color = [self.colors[idx] for idx in idxs]
                    color = ' '.join(color)
                    if color not in colors:
                        colors.append(color)
                while len(animals) < n:
                    idxs = random.sample(range(self.n_symbols), 
                                         self.n_positions)
                    animal = [self.animals[idx] for idx in idxs]
                    animal = ' '.join(animal)
                    if animal not in animals:
                        animals.append(animal)
                assert len(colors) == n
                assert len(animals) == n
            return colors, animals

        if strict:
            done = False
            while not done:
                colors, animals = sample_colored_animals()
                hash_value = hash(tuple(colors + animals))
                if hash_value not in self.hash_values:
                    done = True
                    self.hash_values.append(hash_value)
        else:
            colors, animals = sample_colored_animals()


        # Generate train and test sequences
        if condition == 'aligned':
            train_sequence = self.generate_aligned_sequence(n)
        elif condition == 'misaligned': 
            train_sequence = self.generate_misaligned_sequence(n)
        elif condition == 'blocked':
            train_sequence = self.generate_blocked_sequence(n)
        elif condition == 'interleaved':
            train_sequence = self.generate_interleaved_sequence(n)
        else:
            raise ValueError(f"condition not recognized: {condition}")
        test_sequence = self.generate_test_sequence(n, train_sequence)

        # Flip (x,y) to (y,x)
        if random.random() < self.p_flip_xy:
            train_targets = self.flip_xy(train_sequence)
            test_targets = self.flip_xy(test_sequence)
        else:
            train_targets = train_sequence
            test_targets = test_sequence
        
        # Rotate (x,y)
        if rotated:
            train_targets = self.rotate_xy(train_targets)
            test_targets = self.rotate_xy(test_targets)
        
        # Generate text from sequences
        context = self.generate_context(colors, animals, 
                                        train_sequence, train_targets)
        train_lines = self.generate_test_lines(colors, animals, 
                                               train_sequence, train_targets)
        test_lines = self.generate_test_lines(colors, animals, 
                                              test_sequence, test_targets)
        train_sources = self.add_context(context, train_lines)
        test_sources = self.add_context(context, test_lines)

        # Convert targets (x,y) into strings
        train_targets = [f': {x} {y}' for x,y in train_targets]
        test_targets = [f': {x} {y}' for x,y in test_targets]
        # HACK: include ': ' to ensure that last two tokens will be the same

        # Label each sample according to which block it belongs in
        assert len(train_sources) == 9
        train_tags = 5 * ['trainA'] + 4 * ['trainB']
        assert len(test_sources) == 16
        test_tags = 16 * ['test']

        # Save vocab
        if self.save_vocab is not None:
            for samples in [train_sources, train_targets, 
                            test_sources, test_targets]:
                for toks in samples:
                    for tok in toks.split(' '):
                        if tok not in self.vocab:
                            self.vocab[tok] = self.vocab_idx
                            self.vocab_idx += 1
                
        # Save in dictionary
        episode = {'train_sources': train_sources,
                   'test_sources': test_sources,
                   'train_targets': train_targets,
                   'test_targets': test_targets,
                   'train_tags': train_tags,
                   'test_tags': test_tags}
        
        return episode

    def generate_aligned_sequence(self, n):
        assert n == 5, "n must be 5"
        row = [(0,2), (1,2), (3,2), (4,2)]
        col = [(2,0), (2,1), (2,3), (2,4)]
        shuffled_row = random.sample(row, len(row))
        shuffled_col = random.sample(col, len(col))
        if random.random() < 0.5:
            aligned_sequence = shuffled_row + [(2,2)] + shuffled_col
        else:
            aligned_sequence = shuffled_col + [(2,2)] + shuffled_row
        return aligned_sequence

    def generate_misaligned_sequence(self, n):
        assert n == 5, "n must be 5"
        diag1 = [(0,0), (1,1), (3,3), (4,4)]
        diag2 = [(0,4), (1,3), (3,1), (4,0)]
        shuffled_diag1 = random.sample(diag1, len(diag1))
        shuffled_diag2 = random.sample(diag2, len(diag2))
        if random.random() < 0.5:
            misaligned_sequence = shuffled_diag1 + [(2,2)] + shuffled_diag2
        else:
            misaligned_sequence = shuffled_diag2 + [(2,2)] + shuffled_diag1
        return misaligned_sequence

    def generate_blocked_sequence(self, n):
        # Choose one column and one row to demonstrate
        rand_x = random.randint(0,n-1) # which col to demonstrate
        rand_y = random.randint(0,n-1) # which row to demonstrate

        # Gather remaining rows and columns
        xs = [x for x in range(n)] # possible cols to pair with chosen row 
        xs.remove(rand_x) # remove duplicate
        ys = [y for y in range(n)] # possible rows to pair with chosen col
        ys.remove(rand_y) # remove duplicate

        # Shuffle order of presentation
        shuffled_xs = random.sample(xs, n-1)
        shuffled_ys = random.sample(ys, n-1)
        x_seq = [(rand_x, y) for y in shuffled_ys]
        y_seq = [(x, rand_y) for x in shuffled_xs]

        # Randomize whether column or row is demonstrated first
        if random.random() < 0.5:
            blocked_sequence = x_seq + [(rand_x, rand_y)] + y_seq
        else:
            blocked_sequence = y_seq + [(rand_x, rand_y)] + x_seq
        
        # Check for duplicates
        msg = "Duplicate detected in blocked sequence"
        assert len(blocked_sequence) == len(set(blocked_sequence)), msg

        return blocked_sequence

    def generate_interleaved_sequence(self, n):
        blocked_seq = self.generate_blocked_sequence(n)
        interleaved_sequence = random.sample(blocked_seq, len(blocked_seq))
        return interleaved_sequence

    def generate_test_sequence(self, n, train_sequence):
        all_samples = [(i,j) for i in range(n) for j in range(n)]
        test_sequence = [s for s in all_samples if s not in train_sequence]
        random.shuffle(test_sequence)
        return test_sequence

    def flip_xy(self, sequence):
        flipped = [(y,x) for x,y in sequence]
        return flipped
    
    def generate_context(self, colors, animals, train_sequence, train_targets):
        context = self.preface
        for (color_i, animal_i), (x, y) in zip(train_sequence, train_targets):
            color = colors[color_i]
            animal = animals[animal_i]
            train_sample = self.template
            assert '<color>' in train_sample
            train_sample = train_sample.replace('<color>', color)
            assert '<animal>' in train_sample
            train_sample = train_sample.replace('<animal>', animal)
            assert '<x>' in train_sample
            train_sample = train_sample.replace('<x>', str(x))
            assert '<y>' in train_sample
            train_sample = train_sample.replace('<y>', str(y))
            context += train_sample + self.sep
        return context

    def generate_test_lines(self, colors, animals, test_sequence, test_targets):
        test_samples = []
        for (color_i, animal_i), (x, y) in zip(test_sequence, test_targets):
            color = colors[color_i]
            animal = animals[animal_i]
            test_sample = self.test_preface + self.template
            assert '<color>' in test_sample
            test_sample = test_sample.replace('<color>', color)
            assert '<animal>' in test_sample
            test_sample = test_sample.replace('<animal>', animal)
            assert '<x>' in test_sample
            assert '<y>' in test_sample
            if self.lm_task == 'masked':
                test_sample = test_sample.replace('<x>', '<mask>')
                test_sample = test_sample.replace('<y>', '<mask>')
            elif self.lm_task == 'causal':
                test_sample = test_sample.replace('<x>', str(x))
                test_sample = test_sample.replace('<y>', str(y))
            else: 
                raise ValueError(f"lm_task not recognized: {self.lm_task}")
            test_samples.append(test_sample)
        return test_samples

    def add_context(self, context, test_samples):
        samples = [context + s for s in test_samples]
        return samples
    
    def rotate_xy(self, targets):
        # Make rotation, scaling, and translation matrices
        theta = np.pi/4
        rotate = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        scale = np.eye(2)*(2/np.sqrt(2)) # scale so that all values are integers
        translate = np.array([0,4]) # translate so that all values >= 0
        
        # Transform xy vectors
        new_targets = []
        for x,y in targets:
            vec = np.array([x,y])
            new_vec = vec @ rotate @ scale + translate
            new_x = round(new_vec[0])
            new_y = round(new_vec[1])
            new_targets.append((new_x, new_y))
        
        return new_targets
    
    def make_pretrain(self, episodes, path):
        # Collect episodes and flatten
        srcs = []
        tgts = []
        infos = []
        for split in ['train', 'test']:
            for e_i, episode in enumerate(episodes):
                sources = episode[f'{split}_sources']
                targets = episode[f'{split}_targets']
                # Gather samples
                for s, t in zip(sources, targets):
                    srcs.append(s) # source
                    tgts.append(t) # target

                    # Info
                    info = {'split': split,
                            'episode_id': e_i}
                    infos.append(info)
        
        # Shuffle train and test together for pretraining
        assert len(srcs) == len(tgts)
        assert len(srcs) == len(infos)
        new_ids = [i for i in range(len(srcs))]
        random.shuffle(new_ids)
        shuffled_srcs = [srcs[new_id] for new_id in new_ids]
        shuffled_tgts = [tgts[new_id] for new_id in new_ids]
        shuffled_infos = [infos[new_id] for new_id in new_ids]

        src_fn = path + '.src'
        print(f"Writing data to {src_fn}", flush=True)
        with open(src_fn, 'w') as f:
            for line in shuffled_srcs:
                escaped_line = line.replace('\n', '\\n')  # escape newline chars
                f.write(f'{escaped_line}\n')

        tgt_fn = path + '.tgt'
        print(f"Writing data to {tgt_fn}", flush=True)
        with open(tgt_fn, 'w') as f:
            for line in shuffled_tgts:
                escaped_line = line.replace('\n', '\\n')  # escape newline chars
                f.write(f'{escaped_line}\n')

        info_fn = path + '.info.json'
        print(f"Writing data to {info_fn}", flush=True)
        with open(info_fn, 'w') as f:
            json.dump(shuffled_infos, f) 
    
    def make_finetune(self, episodes, path):
        # Maintain episode structure: finetuning will occur on only one episode
        for e_i, episode in enumerate(episodes):
            # Gather train samples (separated into blocks)
            episode_dict = {'train_all': {'src': [], 'tgt': [], 'info': []},
                            'test_all': {'src': [], 'tgt': [], 'info': []}}
            srcs = episode['train_sources']
            tgts = episode['train_targets']
            tags = episode['train_tags']
            for src, tgt, tag in zip(srcs, tgts, tags):
                if tag not in episode_dict:
                    episode_dict[tag] = {'src': [], 'tgt': [], 'info': []}
                episode_dict[tag]['src'].append(src)
                episode_dict[tag]['tgt'].append(tgt)
                episode_dict[tag]['info'].append({'split': tag,
                                                  'episode_id': e_i})
                assert 'train' in tag
                episode_dict['train_all']['src'].append(src)
                episode_dict['train_all']['tgt'].append(tgt)
                episode_dict['train_all']['info'].append({'split': tag,
                                                        'episode_id': e_i})
            # Gather test samples (separated into blocks)
            srcs = episode['test_sources']
            tgts = episode['test_targets']
            tags = episode['test_tags']
            for src, tgt, tag in zip(srcs, tgts, tags):
                if tag not in episode_dict:
                    episode_dict[tag] = {'src': [], 'tgt': [], 'info': []}
                episode_dict[tag]['src'].append(src)
                episode_dict[tag]['tgt'].append(tgt)
                episode_dict[tag]['info'].append({'split': tag,
                                                  'episode_id': e_i})
                assert 'test' in tag
                episode_dict['test_all']['src'].append(src)
                episode_dict['test_all']['tgt'].append(tgt)
                episode_dict['test_all']['info'].append({'split': tag,
                                                        'episode_id': e_i})
            
            # Save
            episode_dir = os.path.join(path, f'episode{e_i}')
            if not os.path.isdir(episode_dir):
                os.mkdir(episode_dir)
            for split in episode_dict.keys():
                for ext in ['src', 'tgt']:
                    fn = os.path.join(episode_dir, f'{split}.{ext}')
                    with open(fn, 'w') as f:
                        for line in episode_dict[split][ext]:
                            escaped_line = line.replace('\n', '\\n')
                            f.write(f'{escaped_line}\n')
                info_fn = os.path.join(episode_dir, f'{split}.info.json')
                with open(info_fn, 'w') as f:
                    json.dump(episode_dict[split]['info'], f)

class CategoryTask:
    def __init__(self, save_data_to, n_train, n_val, n_test,
                 n_task_dims, n_rel_dims, 
                 n_symbols, n_labels, n_labels_per_task,
                 n_distractors, n_in_context,
                 pt_rot_cond, pt_cond, save_vocab, 
                 preface_id, format, sep_id, test_preface_id,
                 lm_task):
        self.save_data_to = save_data_to # path to directory to save all data
        self.n_train = n_train # number of episodes to use for pretraining
        self.n_val = n_val # number of val episodes to use for pretesting
        self.n_test = n_test # number of episodes to use for finetuning
        self.n_task_dims = n_task_dims # total number of dimensions in task
        self.n_rel_dims = n_rel_dims # number of relevant dimensions per episode
        self.n_symbols = n_symbols # number of symbols per task dimension
        self.n_labels = n_labels # number of total labels (each task uses 4)
        self.n_labels_per_task = n_labels_per_task # number of labels per task
        self.n_distractors = n_distractors # dimensions used for distraction
        self.n_in_context = n_in_context # number of examples to give in context
        self.pt_rot_cond = pt_rot_cond # rotation condition in pretrain phase
        self.pt_cond = pt_cond # condition in pretrain (blocked, interleaved)
        self.save_vocab = save_vocab # path to save vocab to
        self.preface_id = preface_id # which preface to use (prompts.py)
        self.format = format # which format to use for each example
        self.sep_id = sep_id # which sep token to use (prompts.py)
        self.test_preface_id = test_preface_id # which test preface to use
        self.lm_task = lm_task # use masked or causal language modeling task

        # Check that args make sense
        assert n_in_context < n_symbols**2, "n_in_context fewer than total"
        n_total = n_train + n_val + n_test
        n_possible = math.factorial(n_task_dims)/math.factorial(n_task_dims - 2)
        assert n_total <= n_possible, "Not enough possible tasks"
        assert n_labels >= n_labels_per_task, "Not enough labels"

        # Set up vocab, words, etc.
        self.preface = prefaces[preface_id]
        self.sep = seps[sep_id]
        self.test_preface = test_prefaces[test_preface_id]
        if save_vocab is not None:
            self.vocab = {'<pad>': 0}
            self.vocab_idx = 1
        symbols = {}
        for dim in range(self.n_task_dims):
            symbols[f'dim_{dim}'] = []
            for sym in range(self.n_symbols):
                symbols[f'dim_{dim}'].append(f'symbol_{dim}_{sym}')
        labels = [f'label_{i}' for i in range(self.n_labels)]
        self.symbols = symbols
        self.labels = labels

    def make_datasets(self):
        # Generate all tasks (each task is a pair of dimensions)
        all_pairs = [(d1, d2) for d1,d2 in permutations(self.symbols.keys(), 2)]
        n_pairs = len(all_pairs)

        # Generate all labels (each task gets n_labels_per_task random labels)
        all_labels = []
        for _ in range(n_pairs):
            labels = random.sample(self.labels, self.n_labels_per_task)
            if self.n_labels_per_task == 2:
                assert len(labels) == 2
                labels = [labels[0], labels[1], labels[1], labels[0]]
            elif self.n_labels_per_task == 4:
                assert len(labels) == 4
            else:
                raise ValueError(f"n_labels_per_task must be 2 or 4")
            all_labels.append(labels)

        # Shuffle and split into train, val, test
        n_train = self.n_train
        n_val = self.n_val
        n_test = self.n_test
        n_total = n_train + n_val + n_test
        random.shuffle(all_pairs)
        random.shuffle(all_labels)
        train_pairs = all_pairs[:n_train]
        val_pairs = all_pairs[n_train:n_train+n_val]
        test_pairs = all_pairs[n_train+n_val:n_total]
        assert len(train_pairs) == n_train
        assert len(val_pairs) == n_val
        assert len(test_pairs) == n_test
        train_labels = all_labels[:n_train]
        val_labels = all_labels[n_train:n_train+n_val]
        test_labels = all_labels[n_train+n_val:n_total]
        assert len(train_labels) == n_train
        assert len(val_labels) == n_val
        assert len(test_labels) == n_test

        # Generate distractors
        if self.n_distractors > 0:
            all_distractors = []
            for pair in all_pairs:
                irrelevant_dims = set(self.symbols.keys()) - set(pair)
                distractors = random.sample(irrelevant_dims, self.n_distractors)
                all_distractors.append(distractors)
            random.shuffle(all_distractors)
            train_distractors = all_distractors[:n_train]
            val_distractors = all_distractors[n_train:n_train+n_val]
            test_distractors = all_distractors[n_train+n_val:n_total]
        else:
            train_distractors = None
            val_distractors = None
            test_distractors = None

        # Generate episodes for pretraining
        train_episodes = self.generate_episodes(train_pairs, train_labels, 
                                                distractors=train_distractors,
                                                rot_cond=self.pt_rot_cond,
                                                cond=self.pt_cond)
        val_episodes = self.generate_episodes(val_pairs, val_labels,
                                              distractors=val_distractors,
                                              rot_cond=self.pt_rot_cond,
                                              cond=self.pt_cond)
        
        # Generate episodes for finetuning
        unrotated_test = {}
        rotated_test = {}
        for ft_cond in ['blocked', 'interleaved']:
            # Unrotated
            unrotated_eps = self.generate_episodes(test_pairs, test_labels,
                                                   distractors=test_distractors,
                                                   rot_cond='unrotated',
                                                   cond=ft_cond)
            unrotated_test[ft_cond] = unrotated_eps

            # Rotated
            rotated_eps = self.generate_episodes(test_pairs, test_labels,
                                                 distractors=test_distractors,
                                                 rot_cond='rotated',
                                                 cond=ft_cond)
            rotated_test[ft_cond] = rotated_eps

        # Make datasets for pretraining
        pretrain_path = os.path.join(self.save_data_to, 'pretrain')
        self.make_pretrain(train_episodes, pretrain_path)
        pretest_path = os.path.join(self.save_data_to, 'pretest')
        self.make_pretrain(val_episodes, pretest_path)

        # Make datasets for finetuning
        for ft_cond in ['blocked', 'interleaved']:
            # Unrotated
            cond_dir = f'finetune_unrotated_{ft_cond}/'
            unrotated_path = os.path.join(self.save_data_to, cond_dir)
            if not os.path.isdir(unrotated_path):
                os.mkdir(unrotated_path)
            self.make_finetune(unrotated_test[ft_cond], unrotated_path)

            # Rotated
            cond_dir = f'finetune_rotated_{ft_cond}/'
            rotated_path = os.path.join(self.save_data_to, cond_dir)
            if not os.path.isdir(rotated_path):
                os.mkdir(rotated_path)
            self.make_finetune(rotated_test[ft_cond], rotated_path)

        # Vocabulary
        if self.save_vocab is not None:
            print(f"Saving vocab to {self.save_vocab}", flush=True)
            with open(self.save_vocab, 'w') as f:
                json.dump(self.vocab, f)

    def make_pretrain(self, episodes, path):
        # Collect episodes and flatten
        srcs = []
        tgts = []
        infos = []
        for split in ['train', 'test']:
            for e_i, episode in enumerate(episodes):
                sources = episode[f'{split}_sources']
                targets = episode[f'{split}_targets']
                # Gather samples
                for s, t in zip(sources, targets):
                    srcs.append(s) # source
                    tgts.append(t) # target
                
                    # Info
                    info = {'split': split,
                            'episode_id': e_i,
                            'task_dims': episode['task_dims']}
                    infos.append(info)
        
        # Shuffle train and test together for pretraining
        assert len(srcs) == len(tgts)
        assert len(srcs) == len(infos)
        new_ids = [i for i in range(len(srcs))]
        random.shuffle(new_ids)
        shuffled_srcs = [srcs[new_id] for new_id in new_ids]
        shuffled_tgts = [tgts[new_id] for new_id in new_ids]
        shuffled_infos = [infos[new_id] for new_id in new_ids]

        src_fn = path + '.src'
        print(f"Writing data to {src_fn}", flush=True)
        with open(src_fn, 'w') as f:
            for line in shuffled_srcs:
                escaped_line = line.replace('\n', '\\n')
                f.write(f'{escaped_line}\n')
        
        tgt_fn = path + '.tgt'
        print(f"Writing data to {tgt_fn}", flush=True)
        with open(tgt_fn, 'w') as f:
            for line in shuffled_tgts:
                escaped_line = line.replace('\n', '\\n')
                f.write(f'{escaped_line}\n')
        
        info_fn = path + '.info.json'
        print(f"Writing data to {info_fn}", flush=True)
        with open(info_fn, 'w') as f:
            json.dump(shuffled_infos, f)

    def make_finetune(self, episodes, path):
        # Maintain episode structure: finetuning will occur on only one episode
        for e_i, episode in enumerate(episodes):
            # Gather train samples (separated into blocks)
            episode_dict = {'train_all': {'src': [], 'tgt': [], 'info': []},
                            'test_all': {'src': [], 'tgt': [], 'info': []}}
            srcs = episode['train_sources']
            tgts = episode['train_targets']
            tags = episode['train_tags']
            pair = episode['task_dims']
            for src, tgt, tag in zip(srcs, tgts, tags):
                if tag not in episode_dict:
                    episode_dict[tag] = {'src': [], 'tgt': [], 'info': []}
                episode_dict[tag]['src'].append(src)
                episode_dict[tag]['tgt'].append(tgt)
                episode_dict[tag]['info'].append({'split': tag,
                                                  'episode_id': e_i})
                assert 'train' in tag
                episode_dict['train_all']['src'].append(src)
                episode_dict['train_all']['tgt'].append(tgt)
                episode_dict['train_all']['info'].append({'split': tag,
                                                        'episode_id': e_i,
                                                        'task_dims': pair})
            # Gather test samples (separated into blocks)
            srcs = episode['test_sources']
            tgts = episode['test_targets']
            tags = episode['test_tags']
            for src, tgt, tag in zip(srcs, tgts, tags):
                if tag not in episode_dict:
                    episode_dict[tag] = {'src': [], 'tgt': [], 'info': []}
                episode_dict[tag]['src'].append(src)
                episode_dict[tag]['tgt'].append(tgt)
                episode_dict[tag]['info'].append({'split': tag,
                                                  'episode_id': e_i})
                assert 'test' in tag
                episode_dict['test_all']['src'].append(src)
                episode_dict['test_all']['tgt'].append(tgt)
                episode_dict['test_all']['info'].append({'split': tag,
                                                        'episode_id': e_i,
                                                        'task_dims': pair})
            
            # Save
            episode_dir = os.path.join(path, f'episode{e_i}')
            if not os.path.isdir(episode_dir):
                os.mkdir(episode_dir)
            for split in episode_dict.keys():
                for ext in ['src', 'tgt']:
                    fn = os.path.join(episode_dir, f'{split}.{ext}')
                    with open(fn, 'w') as f:
                        for line in episode_dict[split][ext]:
                            escaped_line = line.replace('\n', '\\n')
                            f.write(f'{escaped_line}\n')
                info_fn = os.path.join(episode_dir, f'{split}.info.json')
                with open(info_fn, 'w') as f:
                    json.dump(episode_dict[split]['info'], f)

    def generate_episodes(self, pairs, label_sets, distractors=None, 
                          rot_cond='unrotated', cond='blocked'):
        assert len(pairs) == len(label_sets)
        if distractors is not None:
            assert len(pairs) == len(distractors)
        episodes = []
        for sample_i in range(len(pairs)):
            pair = pairs[sample_i]
            labels = label_sets[sample_i]
            if distractors is not None:
                distractor = distractors[sample_i]

            # Determine random order for symbols + distractors
            example_len = 2 + self.n_distractors
            rand_order = list(range(example_len))
            random.shuffle(rand_order)
            symbols0 = self.symbols[pair[0]]
            symbols1 = self.symbols[pair[1]]

            # Determine which dimension is relevant, if only one is
            if self.n_rel_dims == 1:
                rel_dim = random.choice([0,1])
            elif self.n_rel_dims == 2:
                rel_dim = None
            else:
                n_rel = self.n_rel_dims
                raise ValueError(f"n_rel_dims must be 1 or 2: {n_rel}")

            # Get symbols for distractors
            if distractors is not None:
                distractor_symbols = [self.symbols[d] for d in distractor]

            # Generate all items
            all_items = {}
            n_symbols = self.n_symbols
            for idx0, idx1 in product(range(n_symbols), range(n_symbols)):
                s0, s1 = symbols0[idx0], symbols1[idx1]
                if distractors is not None:
                    ds = [random.choice(d) for d in distractor_symbols]          
                x = idx0 % n_symbols
                y = idx1 % n_symbols

                # Rotate
                if rot_cond == 'rotated':
                    # Rotate by 45 degrees
                    theta = np.pi / 4
                    rotation_matrix = [[np.cos(theta), -np.sin(theta)], 
                                       [np.sin(theta), np.cos(theta)]]
                    rotation_matrix = np.array(rotation_matrix)
                    x, y = np.dot(rotation_matrix, np.array([x, y]))

                    # Shift so everything is positive
                    x += np.sqrt((n_symbols-1)**2/2)

                    # Scale s.t. values are integers (avoid numerical issues)
                    scale = 2*(n_symbols-1) / (2*np.sqrt((n_symbols-1)**2/2))
                    scaling_matrix = np.eye(2) * scale
                    x, y = np.dot(scaling_matrix, np.array([x, y]))
                    x, y = np.round(x), np.round(y)
                    assert x >= 0 and y >= 0, f"x={x}, y={y}"

                    # Determine category boundaries
                    mid = n_symbols - 1
                elif rot_cond == 'unrotated':
                    mid = (n_symbols-1) / 2
                else:
                    raise ValueError(f"Not recognized: {rot_cond}")

                # Determine category labels
                if self.n_rel_dims == 2:
                    cats = [0, 1, 2, 3] # 4 categories even with 2 labels
                    assert n_symbols**2 % 4 == 0
                    if x <= mid and y < mid:
                        category = 0
                    elif x < mid and y >= mid:
                        category = 1
                    elif x > mid and y <= mid:
                        category = 2
                    elif x >= mid and y > mid:
                        category = 3
                    else:
                        raise ValueError("Error assigning label")
                elif self.n_rel_dims == 1:
                    assert self.n_labels_per_task == 2
                    cats = [0, 1, 2, 3] 
                    if rel_dim == 0: # use x
                        if x <= mid and y < mid:
                            # label = labels[0] when n_labels_per_task = 2
                            category = 0 
                        elif x < mid and y >= mid:
                            # label = labels[0] when n_labels_per_task = 2
                            category = 3
                        elif x > mid and y <= mid:
                            # label = labels[1] when n_labels_per_task = 2
                            category = 1
                        elif x >= mid and y > mid:
                            # label = labels[1] when n_labels_per_task = 2
                            category = 2
                    elif rel_dim == 1: # use y
                        if x <= mid and y < mid:
                            # label = labels[0] when n_labels_per_task = 2
                            category = 0 
                        elif x < mid and y >= mid:
                            # label = labels[1] when n_labels_per_task = 2
                            category = 1
                        elif x > mid and y <= mid:
                            # label = labels[0] when n_labels_per_task = 2
                            category = 3
                        elif x >= mid and y > mid:
                            # label = labels[1] when n_labels_per_task = 2
                            category = 2
                    else:
                        raise ValueError("Error assigning label")

                # Add to all_items
                if category not in all_items:
                    all_items[category] = []
                if distractors is not None:
                    all_items[category].append(((s0, s1), ds))
                else:
                    all_items[category].append(((s0, s1)))
                
            
            # Check that all labels have the same number of items
            n_items = [len(all_items[cat]) for cat in cats]
            assert max(n_items) - min(n_items) <= 1

            # Randomly sample items to be given in context
            icl_items = {}
            query_items = {}
            for cat, items in all_items.items():
                msg = "n_in_context must be divisible by 4"
                assert self.n_in_context % 4 == 0, msg
                n_per_label = self.n_in_context // 4
                random.shuffle(items)
                icl_items[cat] = items[:n_per_label] # included in context
                query_items[cat] = items[n_per_label:] # used as test queries
            
            # Generate in-context training sequence
            icl_sequence = []
            random.shuffle(cats)
            for cat in cats: # blocked sequence
                for item in icl_items[cat]:
                    if distractors is not None:
                        s0, s1 = item[0]
                        ds = item[1]
                        example_ = [s0, s1] + ds
                    else:
                        s0, s1 = item
                        example_ = [s0, s1]
                    
                    # Use consistent random order for dims within an episode
                    assert len(example_) == len(rand_order)
                    example = [example_[i] for i in rand_order]

                    label = labels[cat]
                    icl_sequence.append((example, label))
            if cond == 'interleaved':
                random.shuffle(icl_sequence)
            # Generate query sequence (each will be appended to icl sequence)
            query_sequence = []
            for cat in cats:
                for item in query_items[cat]:
                    if distractors is not None:
                        s0, s1 = item[0]
                        ds = item[1]
                        example_ = [s0, s1] + ds
                    else:
                        s0, s1 = item
                        example_ = [s0, s1]
                    
                    # Use consistent random order for dims within an episode
                    assert len(example_) == len(rand_order)
                    example = [example_[i] for i in rand_order]

                    label = labels[cat]
                    query_sequence.append((example, label))
            random.shuffle(query_sequence) # always shuffle queries
            
            # Remove extra info
            icl_sequence = [(item[0], item[1]) for item in icl_sequence]
            query_sequence = [(item[0], item[1]) for item in query_sequence]
            
            # Generate text from sequences
            context = self.generate_context(icl_sequence)
            train_lines = self.generate_test_lines(icl_sequence)
            test_lines = self.generate_test_lines(query_sequence)
            train_sources = self.add_context(context, train_lines)
            test_sources = self.add_context(context, test_lines)

            # Convert labels into targets
            train_targets = [f': {item[1]}' for item in icl_sequence]
            test_targets = [f': {item[1]}' for item in query_sequence]
            # HACK: include ': ' to ensure that last two tokens will be the same

            # Tag each sample according to which block it belongs in
            assert len(train_sources) == self.n_in_context
            train_tags = []
            for example in icl_sequence:
                label = example[1]
                assert label in labels
                tag = f'train{labels.index(label)}'
                train_tags.append(tag)
            assert len(test_sources) == len(query_sequence)
            test_tags = []
            for example in query_sequence:
                label = example[1]
                assert label in labels
                tag = f'test{labels.index(label)}'
                test_tags.append(tag)

            # Save vocab
            if self.save_vocab is not None:
                for samples in [train_sources, train_targets, 
                                test_sources, test_targets]:
                    for toks in samples:
                        for tok in toks.split(' '):
                            if tok not in self.vocab:
                                self.vocab[tok] = self.vocab_idx
                                self.vocab_idx += 1
            
            # Save in dictionary
            episode = {'train_sources': train_sources,
                       'test_sources': test_sources,
                       'train_targets': train_targets,
                       'test_targets': test_targets,
                       'train_tags': train_tags,
                       'test_tags': test_tags,
                       'task_dims': pair} 

            episodes.append(episode)
        return episodes

    def generate_context(self, icl_sequence):
        context = self.preface
        for example, label in icl_sequence:
            if self.format == 'space':
                example_str = ' '.join(example) + ' : ' + label
            elif self.format == 'comma':
                example_str = ' , '.join(example) + ' : ' + label
            elif self.format == 'paren':
                example_str = ' ( ' + ' '.join(example) + ' ) : ' + label
            elif self.format == 'paren_comma':
                example_str = ' ( ' + ' , '.join(example) + ' ) : ' + label
            else:
                raise ValueError(f"Format not recognized: {self.format}")
            context += example_str + self.sep
        return context
    
    def generate_test_lines(self, query_sequence):
        test_lines = []
        for example, label in query_sequence:
            if self.format == 'space':
                example_str = ' '.join(example) + ' : '
            elif self.format == 'comma':
                example_str = ' , '.join(example) + ' : '
            elif self.format == 'paren':
                example_str = ' ( ' + ' '.join(example) + ' ) : '
            elif self.format == 'paren_comma':
                example_str = ' ( ' + ' , '.join(example) + ' ) : '
            else:
                raise ValueError(f"Format not recognized: {self.format}")
            query_str = self.test_preface + example_str
            if self.lm_task == 'masked':
                query_str += '<mask>'
            elif self.lm_task == 'causal':
                query_str += label
            else:
                raise ValueError(f"lm_task not recognized: {self.lm_task}")
            test_lines.append(query_str)
        return test_lines

    def add_context(self, context, test_lines):
        samples = [context + s for s in test_lines]
        return samples
        

class SeqDataset(Dataset):
    """
    Dataset used for meta-learning experiments.
        src [1, src_len]: source (context + test sample)
        tgt [1, tgt_len]: target (xy-coordinates)
        info [dict]: 
            split: 'train' if test_line is in context, else 'test'
    """
    def __init__(self, data_path, vocab, tgt_len, tokenizer=None):
        self.data_path = data_path
        self.vocab = vocab
        self.tgt_len = tgt_len
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.pad_idx = tokenizer.pad_token_id
        else:
            self.pad_idx = vocab['<pad>']

        # Load data
        with open(data_path + '.src', 'r') as f:
            srcs = f.read().splitlines()
            self.srcs = [line.replace('\\n', '\n') for line in srcs]
        with open(data_path + '.tgt', 'r') as f:
            tgts = f.read().splitlines()
            self.tgts = [line.replace('\\n', '\n') for line in tgts]
        with open(data_path + '.info.json') as f:
            self.infos = json.load(f)

        assert len(self.tgts) == len(self.srcs)
        assert len(self.infos) == len(self.srcs)
        self.n_samples = len(self.srcs)

        # Get max length
        self.max_length = None
        max_length = 0
        for src in self.srcs:
            tokens = self.tokenize([src])
            length = tokens['input_ids'].shape[1]
            if length > max_length:
                max_length = length
        self.max_length = max_length


    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, i):
        src = self.srcs[i]
        tgt = self.tgts[i]
        info = self.infos[i] # dictionary
        return src, tgt, info
    
    def tokenize(self, input_strings):
        if self.tokenizer is None:
            # Create batch by tokenizing based on spaces
            batch_ids = []
            for input_string in input_strings:
                input_ids = [self.vocab[tok] for tok in input_string.split(' ')]
                input_ids = torch.tensor(input_ids) # [seq_len]
                batch_ids.append(input_ids)

            # Pad left by flipping, padding right, and then flipping back
            batch_ids = [t.flip(0) for t in batch_ids] # flip
            batch_ids = pad_sequence(batch_ids, batch_first=True, 
                                     padding_value=self.pad_idx) # pad right
            batch_ids = batch_ids.flip(1) # flip back
            # batch_ids: [bs, seq_len]
            
            # Generate attention mask to mask out padding
            attention_mask = (batch_ids != self.pad_idx).type(torch.int64)

            # Return batch in format given by tokenizers
            batch = {'input_ids': batch_ids,
                     'attention_mask': attention_mask}
            
        else:
            batch = self.tokenizer(input_strings, 
                                   padding='max_length',
                                   max_length=self.max_length, 
                                   return_tensors="pt")
        return batch
    
    def collate_fn(self, samples):
        # Unpack
        srcs = [s[0] for s in samples]
        tgts = [s[1] for s in samples]
        infos = [s[2] for s in samples]

        # Create tensors using tokenizer
        src_batch = self.tokenize(srcs) # dict, values : [bs, src_len]
        tgt_batch = self.tokenize(tgts) # dict, values : [bs, tgt_len]
        info_batch = {k:[i[k] for i in infos] for k in infos[0].keys()} # dict

        # HACK: just get last two tokens to avoid colon
        tgt_batch['input_ids'] = tgt_batch['input_ids'][:,-self.tgt_len:] 
        

        return src_batch, tgt_batch, info_batch

def generate_datasets(args):
    # Set up task, build dataset
    assert args.save_data_to is not None
    print("Building task and dataset...", flush=True)
    if args.task == 'grid':
        task = GridTask(args.save_data_to, 
                        args.n_train, args.n_val, args.n_test, 
                        args.n_positions, args.n_symbols, args.pt_rot_cond,
                        args.pt_cond, args.save_vocab, args.p_flip_xy,
                        args.strict_train, args.strict_test, args.generate_all,
                        args.real_words,
                        args.preface_id, args.template_id, 
                        args.sep_id, args.test_preface_id, args.lm_task)
        task.make_datasets()
    elif args.task == 'category':
        task = CategoryTask(args.save_data_to, 
                            args.n_train, args.n_val, args.n_test,
                            args.n_task_dims, args.n_rel_dims,
                            args.n_symbols, args.n_labels,
                            args.n_labels_per_task,
                            args.n_distractors, args.n_in_context,
                            args.pt_rot_cond, args.pt_cond, args.save_vocab,
                            args.preface_id, args.format, args.sep_id,
                            args.test_preface_id, args.lm_task)
        task.make_datasets()
    args.load_data_from = args.save_data_to
    print("Done.", flush=True)

def get_tokenizer(args):
    if args.model_name == 'llama2' and args.llama2_size is not None:
        cache = '/gpfs/data/superlab/models/llama2/llama/checkpoints/hf'
        name = f'Llama-2-{args.llama2_size}-hf'
        path = os.path.join(cache, name)
        tokenizer = LlamaTokenizer.from_pretrained(path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        # Add special token for <mask>
        if args.lm_task == 'masked':
            special_token_dict = {'additional_special_tokens': ['<mask>']}
            tokenizer.add_special_tokens(special_token_dict)
            mask_token_id = tokenizer.convert_tokens_to_ids('<mask>')
            args.mask_idx = mask_token_id
    else:
        tokenizer = None
        if args.lm_task == 'masked':
            args.mask_idx = args.vocab['<mask>']
    return tokenizer

def get_pretrain_loaders(args):
    pretrain_path = os.path.join(args.load_data_from, 'pretrain')
    pretrain_loader = get_loader(pretrain_path, args.pretrain_bs, args)
    pretest_path = os.path.join(args.load_data_from, 'pretest')
    pretest_loader = get_loader(pretest_path, args.pretrain_bs, args)
    return pretrain_loader, pretest_loader

def get_finetune_loaders(args):
    rotated = {}
    unrotated = {}
    if args.task == 'grid':
        ft_conds = ['aligned', 'misaligned', 'blocked', 'interleaved']
    elif args.task == 'category':
        ft_conds = ['blocked', 'interleaved']
    else:
        raise ValueError(f"Task not recognized: {args.task}")
    for ft_cond in ft_conds:
        cond_dir = f'finetune_unrotated_{ft_cond}/'
        unrotated_dir = os.path.join(args.load_data_from, cond_dir)
        cond_dir = f'finetune_rotated_{ft_cond}/'
        rotated_dir = os.path.join(args.load_data_from, cond_dir)
        rotated[ft_cond] = []
        unrotated[ft_cond] = []
        for ep_i in range(args.n_test):
            rotated_ep_path = os.path.join(rotated_dir, f'episode{ep_i}')
            unrotated_ep_path = os.path.join(unrotated_dir, f'episode{ep_i}')
            rotated_episode = {}
            unrotated_episode = {}

            # Get splits
            rotated_splits = []
            for fn in os.listdir(rotated_ep_path):
                if fn.endswith('.src'):
                    rotated_splits.append(fn.split('.')[0])
            unrotated_splits = []
            for fn in os.listdir(unrotated_ep_path):
                if fn.endswith('.src'):
                    unrotated_splits.append(fn.split('.')[0])
            assert set(rotated_splits) == set(unrotated_splits)
            splits = list(set(rotated_splits))

            # Load data from each split
            for split in splits:
                # Rotated
                rotated_path = os.path.join(rotated_ep_path, split)
                rotated_loader = get_loader(rotated_path, args.finetune_bs, 
                                            args)
                rotated_episode[split] = rotated_loader
                # Unrotated
                unrotated_path = os.path.join(unrotated_ep_path, split)
                unrotated_loader = get_loader(unrotated_path, args.finetune_bs, 
                                              args)
                unrotated_episode[split] = unrotated_loader
            rotated[ft_cond].append(rotated_episode)
            unrotated[ft_cond].append(unrotated_episode)
    return rotated, unrotated

def get_loader(data_path, bs, args):

    # Load dataset
    print(f"Loading dataset from {data_path}", flush=True)
    tokenizer = get_tokenizer(args)
    if args.task == 'grid':
        tgt_len = 2
    elif args.task == 'category':
        tgt_len = 1
    dataset = SeqDataset(data_path, args.vocab, tgt_len, tokenizer)
    args.pad_idx = dataset.pad_idx
    print(f"Dataset has {len(dataset)} samples", flush=True)

    # Data loader
    collate_fn = dataset.collate_fn
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, 
                        collate_fn=collate_fn)

    return loader


