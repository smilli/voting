import torch
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass

@dataclass
class ModelData:
  rating_labels: Optional[torch.FloatTensor] = None # rating labels
  user_idxs: Optional[torch.LongTensor] = None
  note_idxs: Optional[torch.LongTensor] = None
  exp_labels: Optional[torch.FloatTensor] = None # exposure labels
  note_features: Optional[torch.FloatTensor] = None

class MatrixFactorizationModel(torch.nn.Module):

    def __init__(self, n_users: int, n_notes: int, 
            exp_user_factors: np.array=None, exp_item_factors: np.array=None, 
            n_components=1):
        super().__init__()
        self.n_users = n_users
        self.n_notes = n_notes

        # embeddings for user and note
        self.user_factors = torch.nn.Embedding(n_users, n_components, sparse=False)
        self.note_factors = torch.nn.Embedding(n_notes, n_components, sparse=False)
        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        torch.nn.init.xavier_uniform_(self.note_factors.weight)
        # standard deviation of uniform RVs used for initialization
        users_std = np.sqrt(6/(n_users+n_components))/np.sqrt(3)
        notes_std = np.sqrt(6/(n_notes+n_components))/np.sqrt(3)

        # embeddings for substitute confounder
        self.use_subconfounder = exp_user_factors is not None
        if exp_user_factors is not None:
          self.confounder_weights = torch.nn.Embedding(n_users, 1, sparse=False)
          self.exp_user_factors = torch.nn.Embedding.from_pretrained(torch.FloatTensor(exp_user_factors))
          # normalize factors to be zero-mean and match the scale of user and note factors
          # normalization doesn't seem to make a huge difference, may remove in ablation later
          self.exp_user_factors.weight.data = self.exp_user_factors.weight.data - torch.mean(self.exp_user_factors.weight)
          self.exp_user_factors.weight.data = self.exp_user_factors.weight.data / torch.std(self.exp_user_factors.weight) * users_std
          # if exp_item_factors is provided, use those weights, otherwise learn them
          if exp_item_factors is not None:
            self.exp_item_factors = torch.nn.Embedding.from_pretrained(torch.FloatTensor(exp_item_factors))
            self.exp_item_factors.weight.data = self.exp_item_factors.weight.data / torch.std(self.exp_item_factors.weight) * notes_std
            self.exp_item_factors.weight.data = self.exp_item_factors.weight.data - torch.mean(self.exp_item_factors.weight)
          else:
            subconf_components = exp_user_factors.shape[1]
            self.exp_item_factors = torch.nn.Embedding(n_notes, subconf_components, sparse=False)
            torch.nn.init.xavier_uniform_(self.exp_item_factors.weight)

        # global intercept
        self.global_intercept = torch.nn.parameter.Parameter(torch.zeros(1, 1))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, data: ModelData):
      """
      Forward pass: get predicted rating for user of note

      Args: data: ModelData object with the following attributes:
          user_idxs (torch.LongTensor): user id, shape (batch_size,)
          note_idxs (torch.LongTensor): note id, shape (batch_size,)
          exposure_labels (torch.FloatTensor): exposure labels, shape (batch_size,)

      Returns:
          torch.FloatTensor: predicted rating, shape (batch_size,)
      """
      treatment_effect = (
        self.user_factors(data.user_idxs) * self.note_factors(data.note_idxs)).sum(
        1, keepdim=True
      )
      if self.use_subconfounder:
        exp_labels = data.exp_labels.unsqueeze(1)
        treatment_effect = treatment_effect * exp_labels
      pred = treatment_effect + self.global_intercept
      if self.use_subconfounder:
        sub_confounder = (
          self.exp_user_factors(data.user_idxs) 
          * self.exp_item_factors(data.note_idxs)).sum(1, keepdim=True)
        sub_confounder = self.confounder_weights(data.user_idxs) * sub_confounder
        pred += sub_confounder
      return pred.squeeze()
    
    def get_votes(self, note_idxs: torch.LongTensor):
      """
      Gets the predicted votes from all users for each note in note_idxs.

      Returns:
        torch.FloatTensor: predicted votes, shape (len(note_idxs), n_users)
      """
      self.eval()
      with torch.no_grad():
        recon_matrix = self.note_factors(note_idxs) @ self.user_factors.weight.T
        recon_matrix += self.global_intercept
        if self.use_subconfounder:
          sub_confounder = self.exp_item_factors(note_idxs) @ self.exp_user_factors.weight.T
          sub_confounder = self.confounder_weights.weight.T * sub_confounder
          recon_matrix += sub_confounder
      return recon_matrix
    
    def get_vote_scores(self, agg_func: Callable, batch_size: int=1024):
      """
      Predicts votes from all users on all notes
      and aggregates predicted votes according to agg_func.
      """
      self.eval()
      vote_scores = []
      with torch.no_grad():
        for start in range(0, self.n_notes, batch_size):
          end = min(start + batch_size, self.n_notes)
          note_idxs = torch.arange(start, end).to(self.device)
          votes = self.get_votes(note_idxs)
          votes = agg_func(votes, dim=1)
          vote_scores.append(votes)
      vote_scores = torch.cat(vote_scores, dim=0)
      return vote_scores
    
    def create_train_val_data(self, data: ModelData, validate_fraction: float = 0.1):
        """
        Create train and validation data

        Args:
            data: ModelData object
            validate_fraction: fraction of data to use for validation

        Returns:
            ModelData: train data
            ModelData: validation data
        """
        n = len(data.rating_labels)
        n_val = int(n * validate_fraction)
        n_train = n - n_val
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        train_idxs = idxs[:n_train]
        val_idxs = idxs[n_train:]
        return ModelData(
          rating_labels=data.rating_labels[train_idxs],
          user_idxs=data.user_idxs[train_idxs],
          note_idxs=data.note_idxs[train_idxs],
          exp_labels=data.exp_labels[train_idxs]
        ), ModelData(
          rating_labels=data.rating_labels[val_idxs],
          user_idxs=data.user_idxs[val_idxs],
          note_idxs=data.note_idxs[val_idxs],
          exp_labels=data.exp_labels[val_idxs]
        )
    
    def weighted_mse_loss(self, data: ModelData, neg_samples_weight=0.3):
      """
      Weighted MSE loss function

      Args: 
        data: ModelData object,
        neg_samples_weight: weight of negative samples

      Returns:
          torch.FloatTensor: loss, shape (1,)
      """
      pred = self.forward(data)
      weights = data.exp_labels + neg_samples_weight * (1 - data.exp_labels)
      loss = torch.nn.functional.mse_loss(pred, data.rating_labels, reduction='none')
      loss = (loss * weights).mean()
      return loss

    def loss(self, data: ModelData, neg_samples_weight=0.3, reg=0.1):
      """
      Loss function: MSE with regularization

      r_{un} - r_hat_{un} + lambda_i * (mu^2 + ||i_u||^2 + ||i_n||^2) + lambda_f * (||f_u||^2 + ||f_n||^2)

      Args: 
        data: ModelData object,
        reg: regularization parameter

      Returns:
          torch.FloatTensor: loss, shape (1,)
      """
      l2_reg_loss = self.weighted_mse_loss(data, neg_samples_weight)
      l2_reg_loss += reg * (
        (self.global_intercept ** 2).mean()
        + (self.user_factors.weight**2).mean()
        + (self.note_factors.weight**2).mean()
      )
      return l2_reg_loss
    
    def get_batch(self, data: ModelData):
      """
      Returns all exposed user-item pairs and a random sample of unexposed user-item pairs.
      The random sample of unexposed pairs is the same size as the exposed pairs.
      """
      num_pos_ratings = len(data.rating_labels)
      users_idx = torch.LongTensor(
        np.random.randint(self.n_users, size=num_pos_ratings)).to(self.device)
      notes_idx = torch.LongTensor(
        np.random.randint(self.n_notes, size=num_pos_ratings)).to(self.device)
      unexp_rating_labels = torch.zeros_like(users_idx, dtype=torch.float).to(self.device)
      curr_rating_labels = torch.cat((data.rating_labels, unexp_rating_labels))
      curr_user_idxs = torch.cat((data.user_idxs, users_idx))
      curr_note_idxs = torch.cat((data.note_idxs, notes_idx))
      curr_exp_labels = torch.cat(
        [
          torch.ones_like(data.rating_labels, dtype=torch.float), 
          torch.zeros_like(users_idx, dtype=torch.float)
        ]).to(self.device)
      curr_data = ModelData(
        rating_labels=curr_rating_labels,
        user_idxs=curr_user_idxs,
        note_idxs=curr_note_idxs,
        exp_labels=curr_exp_labels
      )
      return curr_data

    def fit(self, 
      data: ModelData,
      epochs: int = 10,
      lr: float = 0.1,
      print_interval: int = 1,
      validate_fraction: float = 0.1,
      print_loss: bool = False,
    ):
        """
        Fit model to data

        Args:
            data: ModelData object
            epochs: number of epochs to train for
            lr: learning rate
            print_interval: number of epochs between printing loss
            validate_fraction: fraction of data to use for validation
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        data_train, data_val = self.create_train_val_data(
          data, validate_fraction=validate_fraction)
        train_loss = []
        val_loss = []

        for epoch in range(epochs):
          optimizer.zero_grad()
          batch = self.get_batch(data_train)
          loss = self.loss(batch)
          loss.backward()
          optimizer.step()
          train_loss.append(loss.item())
          if epoch % print_interval == 0:
            val_loss.append(self.loss(data_val).item())
            if print_loss:
              print('Epoch {}: train L2-reg loss = {:.3f}, val L2-reg loss = {:.3f}'.format(
                 epoch, train_loss[-1], val_loss[-1]))
              train_mse = torch.nn.functional.mse_loss(
                self.forward(data_train), data_train.rating_labels)
              val_mse = torch.nn.functional.mse_loss(
                self.forward(data_val), data_val.rating_labels)
              print('Epoch {}: train MSE = {:.3f}, val MSE = {:.3f}'.format(epoch, train_mse, val_mse))
        
        return train_loss, val_loss