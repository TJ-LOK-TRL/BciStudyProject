import numpy as np
from src.datasets.motor_imagery.bci_comp_iv_2a import BCICompIV2a
from src.models.csp_lda import CSPLDAModel
from src.evaluation.validation import evaluate_intra_subject_fixed_split
from src.evaluation.reporting import generate_report
from src.utils.setup_seed import set_global_seed

if __name__ == '__main__':
    set_global_seed()

    print('Loading dataset...')
    dataset = BCICompIV2a(subject_ids=[1], tmin=0.5, tmax=3.5)
    X, y = dataset.get_data()
    subject_ids = dataset.subject_ids_array

    print(f'X: {X.shape}, y: {np.unique(y)}')

    model = CSPLDAModel(n_components=4)

    result = evaluate_intra_subject_fixed_split(
        model=model,
        X=X,
        y=y,
        subject_ids=subject_ids,
        test_ratio=0.2,
        validation_ratio=0.0,
    )
    print(result)

    generate_report(
        result=result,
        dataset=dataset,    
        model=model,
        notes='Test dataset',
    )