def create_model(args):

    if args.phase == 'train':
        from model.train_model import TheModel

    elif args.phase == 'test':
        from model.test_model import TheModel

    model = TheModel(args)
    print("The model has been created.")
    return model
