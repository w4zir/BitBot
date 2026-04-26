$ErrorActionPreference = "Stop"

# Continue fine-tuning ModernBERT on simulated_errors.
# Edit the variables below, then run:
#   powershell -ExecutionPolicy Bypass -File .\training\experiments\run_multiclass_continue_finetune.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

# -------- configurable paths --------
$TrainScript = Join-Path $RepoRoot "training\experiments\src\train_multiclass_modernbert.py"
$DatasetDir = Join-Path $RepoRoot "training\data\simulated_errors"
$TrainFile = Join-Path $DatasetDir "train.jsonl"

# Since this dataset currently has only train.jsonl, we reuse it for eval/test.
# Replace these with dedicated files if/when you add them.
$EvalFile = Join-Path $DatasetDir "train.jsonl"
$TestFile = Join-Path $DatasetDir "train.jsonl"

# Must match the label IDs used in your dataset.
$Label2IdFile = Join-Path $RepoRoot "training\data\bitext_category\label2id.json"

# Point this to your previously fine-tuned checkpoint directory
# (it should contain config.json + model weights).
$LocalBaseModelDir = Join-Path $RepoRoot "training\models\bitext_multiclass_finetuned_20260415T052123Z\winner"

# Output folder base for new run artifacts.
$OutputDir = Join-Path $RepoRoot "training\models\simulated_errors_multiclass_continue"

# -------- training knobs --------
$NumEpochs = 10
$BatchSize = 16
$EvalBatchSize = 8
$LearningRate = 5e-5
$EvalSteps = 50

python $TrainScript `
  --dataset-dir $DatasetDir `
  --train-file $TrainFile `
  --eval-file $EvalFile `
  --test-file $TestFile `
  --label2id-file $Label2IdFile `
  --local-base-model-dir $LocalBaseModelDir `
  --output-dir $OutputDir `
  --num-epochs $NumEpochs `
  --batch-size $BatchSize `
  --eval-batch-size $EvalBatchSize `
  --learning-rate $LearningRate `
  --eval-steps $EvalSteps
