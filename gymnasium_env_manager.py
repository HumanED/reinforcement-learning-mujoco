import gymnasium
import pandas as pd

while True:
    print("list:        lists registered environments")
    print("delete env:  deletes env from environments")
    print("exit:        exit application")
    command = input("")
    if command == "list":
        all_envs = gymnasium.envs.registry.keys()
        for env_id in all_envs:
            print(env_id)
        print("DUPLICATES")
        unique = set()
        for env_id in all_envs:
            if env_id in unique:
                print(env_id)
            unique.add(env_id)

    elif command.startswith("delete"):
        kill = [
    "FetchSlide-v1", "FetchSlide-v4", "FetchPickAndPlace-v1", "FetchPickAndPlace-v4",
    "FetchReach-v1", "FetchReach-v4", "FetchPush-v1", "FetchPush-v4",
    "HandReach-v0", "HandReach-v3", "HandManipulateBlockRotateZ-v0", "HandManipulateBlockRotateZ-v1",
    "HandManipulateBlockRotateZ_BooleanTouchSensors-v0", "HandManipulateBlockRotateZ_BooleanTouchSensors-v1",
    "HandManipulateBlockRotateZ_ContinuousTouchSensors-v0", "HandManipulateBlockRotateZ_ContinuousTouchSensors-v1",
    "HandManipulateBlockRotateParallel-v0", "HandManipulateBlockRotateParallel-v1",
    "HandManipulateBlockRotateParallel_BooleanTouchSensors-v0", "HandManipulateBlockRotateParallel_BooleanTouchSensors-v1",
    "HandManipulateBlockRotateParallel_ContinuousTouchSensors-v0", "HandManipulateBlockRotateParallel_ContinuousTouchSensors-v1",
    "HandManipulateBlockRotateXYZ-v0", "HandManipulateBlockRotateXYZ-v1",
    "HandManipulateBlockRotateXYZ_BooleanTouchSensors-v0", "HandManipulateBlockRotateXYZ_BooleanTouchSensors-v1",
    "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v0", "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1",
    "HandManipulateBlockFull-v0", "HandManipulateBlockFull-v1", "HandManipulateBlock-v0", "HandManipulateBlock-v1",
    "HandManipulateBlock_BooleanTouchSensors-v0", "HandManipulateBlock_BooleanTouchSensors-v1",
    "HandManipulateBlock_ContinuousTouchSensors-v0", "HandManipulateBlock_ContinuousTouchSensors-v1",
    "HandManipulateEggRotate-v0", "HandManipulateEggRotate-v1", "HandManipulateEggRotate_BooleanTouchSensors-v0",
    "HandManipulateEggRotate_BooleanTouchSensors-v1", "HandManipulateEggRotate_ContinuousTouchSensors-v0",
    "HandManipulateEggRotate_ContinuousTouchSensors-v1", "HandManipulateEggFull-v0", "HandManipulateEggFull-v1",
    "HandManipulateEgg-v0", "HandManipulateEgg-v1", "HandManipulateEgg_BooleanTouchSensors-v0",
    "HandManipulateEgg_BooleanTouchSensors-v1", "HandManipulateEgg_ContinuousTouchSensors-v0",
    "HandManipulateEgg_ContinuousTouchSensors-v1", "HandManipulatePenRotate-v0", "HandManipulatePenRotate-v1",
    "HandManipulatePenRotate_BooleanTouchSensors-v0", "HandManipulatePenRotate_BooleanTouchSensors-v1",
    "HandManipulatePenRotate_ContinuousTouchSensors-v0", "HandManipulatePenRotate_ContinuousTouchSensors-v1",
    "HandManipulatePenFull-v0", "HandManipulatePenFull-v1", "HandManipulatePen-v0", "HandManipulatePen-v1",
    "HandManipulatePen_BooleanTouchSensors-v0", "HandManipulatePen_BooleanTouchSensors-v1",
    "HandManipulatePen_ContinuousTouchSensors-v0", "HandManipulatePen_ContinuousTouchSensors-v1",
    "AntMaze_UMaze-v3", "AntMaze_Open-v3", "AntMaze_Open_Diverse_G-v3", "AntMaze_Open_Diverse_GR-v3",
    "AntMaze_Medium-v3", "AntMaze_Medium_Diverse_G-v3", "AntMaze_Medium_Diverse_GR-v3",
    "AntMaze_Large-v3", "AntMaze_Large_Diverse_G-v3", "AntMaze_Large_Diverse_GR-v3",
    "AntMaze_UMaze-v4", "AntMaze_Open-v4", "AntMaze_Open_Diverse_G-v4", "AntMaze_Open_Diverse_GR-v4",
    "AntMaze_Medium-v4", "AntMaze_Medium_Diverse_G-v4", "AntMaze_Medium_Diverse_GR-v4",
    "AntMaze_Large-v4", "AntMaze_Large_Diverse_G-v4", "AntMaze_Large_Diverse_GR-v4",
    "AntMaze_UMaze-v5", "AntMaze_Open-v5", "AntMaze_Open_Diverse_G-v5", "AntMaze_Open_Diverse_GR-v5",
    "AntMaze_Medium-v5", "AntMaze_Medium_Diverse_G-v5", "AntMaze_Medium_Diverse_GR-v5",
    "AntMaze_Large-v5", "AntMaze_Large_Diverse_G-v5", "AntMaze_Large_Diverse_GR-v5",
    "PointMaze_UMaze-v3", "PointMaze_Open-v3", "PointMaze_Open_Diverse_G-v3", "PointMaze_Open_Diverse_GR-v3",
    "PointMaze_Medium-v3", "PointMaze_Medium_Diverse_G-v3", "PointMaze_Medium_Diverse_GR-v3",
    "PointMaze_Large-v3", "PointMaze_Large_Diverse_G-v3", "PointMaze_Large_Diverse_GR-v3",
    "FetchSlideDense-v1", "FetchSlideDense-v4", "FetchPickAndPlaceDense-v1", "FetchPickAndPlaceDense-v4",
    "FetchReachDense-v1", "FetchReachDense-v4", "FetchPushDense-v1", "FetchPushDense-v4",
    "HandReachDense-v0", "HandReachDense-v3", "HandManipulateBlockRotateZDense-v0", "HandManipulateBlockRotateZDense-v1",
    "HandManipulateBlockRotateZ_BooleanTouchSensorsDense-v0", "HandManipulateBlockRotateZ_BooleanTouchSensorsDense-v1",
    "HandManipulateBlockRotateZ_ContinuousTouchSensorsDense-v0", "HandManipulateBlockRotateZ_ContinuousTouchSensorsDense-v1",
    "HandManipulateBlockRotateParallelDense-v0", "HandManipulateBlockRotateParallelDense-v1",
    "HandManipulateBlockRotateParallel_BooleanTouchSensorsDense-v0", "HandManipulateBlockRotateParallel_BooleanTouchSensorsDense-v1",
    "HandManipulateBlockRotateParallel_ContinuousTouchSensorsDense-v0", "HandManipulateBlockRotateParallel_ContinuousTouchSensorsDense-v1",
    "HandManipulateBlockRotateXYZDense-v0", "HandManipulateBlockRotateXYZDense-v1",
    "HandManipulateBlockRotateXYZ_BooleanTouchSensorsDense-v0", "HandManipulateBlockRotateXYZ_BooleanTouchSensorsDense-v1",
    "HandManipulateBlockRotateXYZ_ContinuousTouchSensorsDense-v0", "HandManipulateBlockRotateXYZ_ContinuousTouchSensorsDense-v1",
    "HandManipulateBlockFullDense-v0", "HandManipulateBlockFullDense-v1", "HandManipulateBlockDense-v0", "HandManipulateBlockDense-v1",
    "HandManipulateBlock_BooleanTouchSensorsDense-v0", "HandManipulateBlock_BooleanTouchSensorsDense-v1",
    "HandManipulateBlock_ContinuousTouchSensorsDense-v0", "HandManipulateBlock_ContinuousTouchSensorsDense-v1",
    "HandManipulateEggRotateDense-v0", "HandManipulateEggRotateDense-v1", "HandManipulateEggRotate_BooleanTouchSensorsDense-v0",
    "HandManipulateEggRotate_BooleanTouchSensorsDense-v1", "HandManipulateEggRotate_ContinuousTouchSensorsDense-v0",
    "HandManipulateEggRotate_ContinuousTouchSensorsDense-v1", "HandManipulateEggFullDense-v0", "HandManipulateEggFullDense-v1",
    "HandManipulateEggDense-v0", "HandManipulateEggDense-v1", "HandManipulateEgg_BooleanTouchSensorsDense-v0",
    "HandManipulateEgg_BooleanTouchSensorsDense-v1", "HandManipulateEgg_ContinuousTouchSensorsDense-v0",
    "HandManipulateEgg_ContinuousTouchSensorsDense-v1", "HandManipulatePenRotateDense-v0", "HandManipulatePenRotateDense-v1",
    "HandManipulatePenRotate_BooleanTouchSensorsDense-v0", "HandManipulatePenRotate_BooleanTouchSensorsDense-v1",
    "HandManipulatePenRotate_ContinuousTouchSensorsDense-v0", "HandManipulatePenRotate_ContinuousTouchSensorsDense-v1",
    "HandManipulatePenFullDense-v0", "HandManipulatePenFullDense-v1", "HandManipulatePenDense-v0", "HandManipulatePenDense-v1",
    "HandManipulatePen_BooleanTouchSensorsDense-v0", "HandManipulatePen_BooleanTouchSensorsDense-v1",
    "HandManipulatePen_ContinuousTouchSensorsDense-v0", "HandManipulatePen_ContinuousTouchSensorsDense-v1",
    "AntMaze_UMazeDense-v3", "AntMaze_OpenDense-v3", "AntMaze_Open_Diverse_GDense-v3", "AntMaze_Open_Diverse_GRDense-v3",
    "AntMaze_MediumDense-v3", "AntMaze_Medium_Diverse_GDense-v3", "AntMaze_Medium_Diverse_GRDense-v3",
    "AntMaze_LargeDense-v3", "AntMaze_Large_Diverse_GDense-v3", "AntMaze_Large_Diverse_GRDense-v3",
    "AntMaze_UMazeDense-v4", "AntMaze_OpenDense-v4", "AntMaze_Open_Diverse_GDense-v4", "AntMaze_Open_Diverse_GRDense-v4",
    "AntMaze_MediumDense-v4", "AntMaze_Medium_Diverse_GDense-v4", "AntMaze_Medium_Diverse_GRDense-v4",
    "AntMaze_LargeDense-v4", "AntMaze_Large_Diverse_GDense-v4", "AntMaze_Large_Diverse_GRDense-v4",
    "AntMaze_UMazeDense-v5", "AntMaze_OpenDense-v5", "AntMaze_Open_Diverse_GDense-v5", "AntMaze_Open_Diverse_GRDense-v5",
    "AntMaze_MediumDense-v5", "AntMaze_Medium_Diverse_GDense-v5", "AntMaze_Medium_Diverse_GRDense-v5",
    "AntMaze_LargeDense-v5", "AntMaze_Large_Diverse_GDense-v5", "AntMaze_Large_Diverse_GRDense-v5",
    "PointMaze_UMazeDense-v3", "PointMaze_OpenDense-v3", "PointMaze_Open_Diverse_GDense-v3", "PointMaze_Open_Diverse_GRDense-v3",
    "PointMaze_MediumDense-v3", "PointMaze_Medium_Diverse_GDense-v3", "PointMaze_Medium_Diverse_GRDense-v3",
    "PointMaze_LargeDense-v3", "PointMaze_Large_Diverse_GDense-v3", "PointMaze_Large_Diverse_GRDense-v3",
    "AdroitHandDoorSparse-v1", "AdroitHandHammerSparse-v1", "AdroitHandPenSparse-v1", "AdroitHandRelocateSparse-v1",
    "AdroitHandDoor-v1", "AdroitHandHammer-v1", "AdroitHandPen-v1", "AdroitHandRelocate-v1",
    "FrankaKitchen-v1"
]
        for env_id in kill:
            if env_id in gymnasium.envs.registration.registry:
                del gymnasium.envs.registration.registry[env_id]
                # del gymnasium.envs.registry[env_id]
        print("DONE")
        # else:
        #     print("Environment does not exist")
    else:
        break
