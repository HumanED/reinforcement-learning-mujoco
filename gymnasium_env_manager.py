import gymnasium

while True:
    print("list:        lists registered environments")
    print("delete env:  deletes env from environments")
    print("exit:        exit application")
    command = input("")
    if command == "list":

        # all_envs = gymnasium.envs.registry.keys()
        all_envs = gymnasium.envs.registration.registry.keys()
        for env_id in all_envs:
            print(env_id)
        print("DUPLICATES")
        unique = set()
        for env_id in all_envs:
            if env_id in unique:
                print(env_id)
            unique.add(env_id)

    elif command.startswith("delete"):
        env_id = command.split(" ")[1]
        if env_id in gymnasium.envs.registration.registry:
            del gymnasium.envs.registration.registry[env_id]
                # del gymnasium.envs.registry[env_id]
            print("DONE")
        else:
            print("NOT FOUND")
        # else:
        #     print("Environment does not exist")
    else:
        break
