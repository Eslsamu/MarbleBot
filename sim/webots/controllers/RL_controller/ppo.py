from webots.controllers.RL_controller.wbt_jobs import run_job
from tensorflow.saved_model import simple_save

SAVE_MODEL_PATH = "webots/controllers/RL_controller/saved_model"

def save_model(sess, inputs, outputs, export_dir = SAVE_MODEL_PATH):
    simple_save(
        sess,
        export_dir,
        inputs,
        outputs
    )

save_model()
run_job(build_files=True)


