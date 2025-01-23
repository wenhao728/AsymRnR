import re
from typing import List, Union


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def calculate_batch_size(
    text_prompts: Union[str, List[str]],
    cfg_scale: float = 1.0,
):
    if isinstance(text_prompts, str):
        batch_zie = 1
    else:
        batch_zie = len(text_prompts)

    if cfg_scale > 1:
        batch_zie *= 2
    return batch_zie


def prompt_to_file_name(text_prompt: str, prefix=None, suffix=None, max_str_len=20) -> str:
    file_name = re.sub(r'[^\w\s]', '', text_prompt).strip()
    if len(file_name) > max_str_len:
        file_name = file_name[:max_str_len]
    if prefix is not None:
        file_name = f'{prefix:03d}-{file_name}'
    if suffix is not None:
        file_name += f'-{suffix:02d}'
    return file_name