# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'SHHA':
        from crowd_datasets.SHHA.loading_data import loading_data
        return loading_data
        
    elif args.dataset_file == 'Wildlife':
        from crowd_datasets.Wildlife.loading_data import loading_data
        return loading_data

    elif args.dataset_file == 'SHAA_edited':
        from crowd_datasets.SHAA_edited.loading_data import loading_data
        return loading_data

    return None