from egomogen.utils.body_model import BodyModelSMPLX, BodyModelSMPLH


def make_smplx(type, **kwargs,):
    if type == 'trumans':
        model = BodyModelSMPLX(
            model_path='EgoMoGen_data/SMPL_Models',
            model_type='smplx',
            gender=kwargs.get('gender', 'male'),
            ext='npz',
            num_betas=10,
            use_pca=False,
        )
    elif type == 'egobody':
        model = BodyModelSMPLX(
            model_path='EgoMoGen_data/SMPL_Models',
            model_type='smplx',
            gender=kwargs.get('gender', 'male'),
            ext='npz',
            num_pca_comps=12,
        )
    elif type == 'boxing':
        model = BodyModelSMPLX(model_path='EgoMoGen_data/SMPL_Models',
                          model_type='smplx',
                          gender="male",
                          ext='npz',
                          use_pca=False,
                          )
    else:
        raise NotImplementedError
    
    return model