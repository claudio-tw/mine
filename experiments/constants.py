import torch


# The following parameters, when 
# passed to tests.normal.multivariates, 
# give an error in the out-of-sample i
# estimated mutual information 
# smaller than 1.7%
normal = dict(
    dim_x = 1,
    dim_y = 1,
    number_of_samples = int(2.5e6),
    empirical_sample_size = 10000,
    learning_rate = 2e-3,
    batch_size = 1000,
    num_of_epochs = 7,
    device = torch.device('cpu'),
)



# The following parameters, when 
# passed to tests.normal.smile, 
# give a good match of
# estimated mutual information 
# with the thoretical quantity
normal_smile = dict(
    grid_size = 7,
    number_of_samples = int(2.5e6),
    insample_empirical_sample_size = 10000,
    outsample_empirical_sample_size = 10000,
    learning_rate = 5e-3,
    batch_size = 1000,
    num_of_epochs = 6,
    device = torch.device('cpu'),
)




# The following parameters, when 
# passed to tests.normal.smile, 
# give a good inequality
# between mutual information of 
# all noise projection
# and mutual information of
# zero-noise projection
projections = dict(
    dim_x = 3,
    dim_y = 2,
    number_of_samples = int(2.5e6),
    empirical_sample_size = 7500,
    batch_size = 1000,
    learning_rate = 5e-3,
    num_of_epochs = 3,
)
