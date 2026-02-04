_base_ = ['../../_base_/schedules/classification.py', '../../_base_/default_runtime.py']

# transform
train_pipeline = [
    dict(type="SmileToGraph", keys=["input"]),
    dict(type="DGLGraphToPyG", keys=["input"], node_feat_key="x", edge_feat_key="x"),
    dict(type="PackPyG", graph_key="input", label_key="gt_label", group_key="group"),
]
test_pipeline = [
    dict(type="SmileToGraph", keys=["input"]),
    dict(type="DGLGraphToPyG", keys=["input"], node_feat_key="x", edge_feat_key="x"),
    dict(type="PackPyG", graph_key="input", label_key="gt_label", group_key="group"),
]

# dataset
dataset_type = "LBAPDataset"
ann_file = '/workspace/DrugOOD/data/lbap_core_ec50_assay.json'

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        split="train",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline
    ),
    val=dict(
        split="ood_val",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        rule="greater",
        save_best="auc"
    ),
    test=dict(
        split="ood_test",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
    ),
)

# model
model = dict(num_classes=2)
