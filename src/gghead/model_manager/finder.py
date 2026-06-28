from elias.folder import ModelFolder
from elias.manager import ModelManager

from src.gghead.model_manager.gghead_model_manager import GGHeadModelFolder
from src.gghead.model_manager.dgghead_model_manager import DGGHeadModelFolder



_RUN_NAME_TO_MODEL_FOLDER_CLASS = {
    'GGHEAD' : GGHeadModelFolder,
    'DGGHEAD' : DGGHeadModelFolder
}

def find_model_folder(run_name: str) -> ModelFolder:
    # for model_folder_cls in _MODEL_FOLDERS_CLASSES:
    model_folder_cls = _RUN_NAME_TO_MODEL_FOLDER_CLASS[run_name.split('-')[0]]
    model_folder = model_folder_cls()

    if model_folder.resolve_run_name(run_name) is not None:
        return model_folder

    raise ValueError(f"Could not locate model folder for run {run_name}. Is the run name correct?")


def find_model_manager(run_name: str) -> ModelManager:
    model_folder = find_model_folder(run_name)
    return model_folder.open_run(run_name)