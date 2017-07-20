import pytest
import unittest.mock as mock

import open_cp.gui.hierarchical as hier

def test_model():
    model = hier.Model(5)
    assert model.number_keys == 5
    with pytest.raises(AttributeError):
        model.number_keys = 7
    
@pytest.fixture
def dict_model_1():
    return hier.DictionaryModel({
        (1,2,3) : 5,
        (1,2,4) : 7
            })

def test_model_current_selection(dict_model_1):
    with pytest.raises(ValueError):
        dict_model_1.current_selection = (1,2)

    dict_model_1.current_selection = (1,2,3)
    assert dict_model_1.current_selection == (1,2,3)

def test_dict_model(dict_model_1):
    dm = dict_model_1

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
    
def test_dict_model_invalid_key(dict_model_1):
    with pytest.raises(KeyError):
        dict_model_1.get((1,2,5))

def test_hierarchical_init(dict_model_1):
    view = mock.Mock()
    cont = hier.Hierarchical(dict_model_1, view)
    assert view.set_choices.call_args_list == [mock.call(0, [1]),
            mock.call(1, [2]), mock.call(2,[3,4])]
    assert view.set_selection.call_args_list == [mock.call(0, 1),
            mock.call(1, 2), mock.call(2, 3)]
    assert dict_model_1.current_selection == (1,2,3)
    assert dict_model_1.current_item == 5

@pytest.fixture
def dict_model_2():
    return hier.DictionaryModel({
        (1,2,3) : 5,
        (1,2,4) : 7,
        (1,3,2) : 8,
        (2,1,1,) : 9
            })

def test_hierarchical_new_selection(dict_model_2):
    view = mock.Mock()
    cont = hier.Hierarchical(dict_model_2, view)
    view.reset_mock()
    cont.new_selection(0, 1)
    assert view.set_choices.call_args_list == [mock.call(1, [2, 3]),
            mock.call(2, [3, 4])]
    assert view.set_selection.call_args_list == [mock.call(1, 2), mock.call(2, 3)]
    assert dict_model_2.current_selection == (1,2,3)
    assert dict_model_2.current_item == 5

    view.reset_mock()
    cont.new_selection(1, 3)
    assert view.set_choices.call_args_list == [mock.call(2, [2])]
    assert view.set_selection.call_args_list == [mock.call(2, 2)]
    assert dict_model_2.current_selection == (1,3,2)
    assert dict_model_2.current_item == 8

    view.reset_mock()
    cont.callback = mock.Mock()
    cont.new_selection(0, 2)
    assert view.set_choices.call_args_list == [mock.call(1, [1]), mock.call(2, [1])]
    assert view.set_selection.call_args_list == [mock.call(1, 1), mock.call(2, 1)]
    assert dict_model_2.current_selection == (2,1,1)
    assert dict_model_2.current_item == 9
    assert cont.callback.called
