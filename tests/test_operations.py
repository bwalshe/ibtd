import pytest
from ibtd.operations import *


def idfn(x):
    return x


def test_op_constructor():
    expected_name = "test"
    expected_desc = "This is a test"
    test_op = Op(expected_name, expected_desc, idfn, (idfn,))
    assert test_op.name == expected_name
    assert test_op.description == expected_desc
    with pytest.raises(Exception):
        bad_op = Op(expected_name, expected_desc, idfn, (idfn, idfn))


def test_op_table():
    expected_desc = { "A":"A", "B":"B" }
    ops = [Op(name, desc, idfn, (idfn,)) for name, desc in expected_desc.items()]
    table = OpTable(*ops)
    assert len(table) == len(expected_desc)
    assert expected_desc == table.op_descriptions()