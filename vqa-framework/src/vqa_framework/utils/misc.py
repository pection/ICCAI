import os
from git import Repo
from typing import List, Callable, Dict, Optional
import inspect
from argparse import ArgumentParser
import torch


def strtobool(x: str):
    if x.lower() in ["y", "yes", "t", "true", "on", "1"]:
        return True
    elif x.lower() in ["n", "no", "f", "false", "off", "0"]:
        return False
    else:
        raise ValueError(f"Cannot parse {x} as True or False")


def update_dict(old_dict: Dict, new_vals: Dict) -> None:
    """
    Mutate old_dict so that it now contains all key-value pairs in new_vals.
    Any pre-existing keys are overwritten
    """
    for key in new_vals:
        old_dict[key] = new_vals[key]


def calculate_acc_from_logits(logits: torch.Tensor, answers: torch.LongTensor) -> float:
    """

    :param logits: NxA tensor where A is the number of answers
    :param answers: A tensor where each element is the answer index
    :return:
    """
    return torch.mean((torch.argmax(logits, dim=1) == answers).float()).item()


def uncommited_changes(path_to_root="."):
    repo = Repo(path_to_root)
    assert not repo.bare
    return repo.is_dirty() or len(repo.untracked_files) >= 1


def checkout_repo(
    url: str, target_dir: str, target_name: str, commit: Optional[str] = None
):
    """
    Checkout <url> at commit <commit> (or most recent if None), and save
    to <target_dir> with name, <target_name>.
    """
    Repo.clone_from(url, os.path.join(target_dir, target_name))
    repo = Repo(os.path.join(target_dir, target_name))
    if commit is not None:
        repo.git.reset("--hard", commit)
    assert not repo.bare


def add_init_args(
    parser: ArgumentParser,
    initializer: Callable,
    exclusions: List[str] = [],
    prefix: str = "",
    postfix: str = "",
    defaults: Dict[str, any] = {},
    inclusions: Optional[List[str]] = None,
):
    """
    Function that takes all the arguments in the initializer <initializer>,
    and adds them into the <parser>, unless the argument is on the <exclusions> list.
    Any default values will be transfered, and all arguments *must* have a type
    annotation.

    Any existing defaults will be overrided by <defaults> if present

    :param parser: Argument parser to update
    :param initializer: The __init__ function of the class being added
    :param exclusions: Any arguments to exclude
    :param prefix: A string prefix to add to all arg names
    :param postfix: A string postfix to add to all arg names
    :param inclusions: If not None, then only add these arguments
    :return:
    """
    signature = inspect.signature(initializer)
    for p_name in signature.parameters:
        if p_name not in exclusions + ["self", "kwargs"] and (
            inclusions is None or p_name in inclusions
        ):
            param = signature.parameters[p_name]
            assert param.annotation != inspect.Parameter.empty
            # arg_parse_args = {"name_or_flags": f"--{prefix}{p_name}{postfix}"}
            arg_parse_args = {"required": param.default == inspect.Parameter.empty}
            if param.default != inspect.Parameter.empty:
                arg_parse_args["default"] = defaults.get(p_name, param.default)
            if param.annotation == bool:
                # Hack to get boolean args to work
                arg_parse_args["type"] = lambda x: bool(strtobool(str(x)))
            # Hack to try and deal with (most) optional arguments
            elif str(param.annotation) in [
                "typing.Union[int, NoneType]",
                "typing.Optional[int]",
            ]:
                arg_parse_args["type"] = int
            elif str(param.annotation) in [
                "typing.Union[str, NoneType]",
                "typing.Optional[str]",
            ]:
                arg_parse_args["type"] = str
            elif str(param.annotation) in [
                "typing.Union[float, NoneType]",
                "typing.Optional[float]",
            ]:
                arg_parse_args["type"] = float
            else:
                # Pray it works
                arg_parse_args["type"] = param.annotation
            print(p_name)
            print(arg_parse_args)
            parser.add_argument(f"--{prefix}{p_name}{postfix}", **arg_parse_args)
