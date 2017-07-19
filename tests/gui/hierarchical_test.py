import pytest

import open_cp.gui.hierarchical as hier

def test_model():
    model = hier.Model(5)
    assert model.number_keys == 5
    with pytest.raises(AttributeError):
        model.number_keys = 7
    
def test_dict_model():
    dm = hier.DictionaryModel({
        (1,2,3) : 5,
        (1,2,4) : 7
            })

    assert dm.get((1,2,3)) == 5
    assert dm.get((1,2,4)) == 7
    
    assert set(dm.get_key_options(())) == {1}
    assert set(dm.get_key_options( (5,) )) == set()
    assert set(dm.get_key_options( (1,) )) == {2}
    assert set(dm.get_key_options( (1,2) )) == {4,3}
    
def test_dict_model_not_tuples():
    with pytest.raises(ValueError):
        hier.DictionaryModel({5:7})
        
def test_dict_model_keys_different_lengths():
    with pytest.raises(ValueError):
        hier.DictionaryModel({(1,2):5, (1,2,3) : 7})
    
