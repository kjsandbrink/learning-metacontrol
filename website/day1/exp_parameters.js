function returnExperimentParameters() {
    let params = {

        // General parameters
        task: task_name,
        id: jsPsych.randomization.randomID(10),
        group: Math.random() >= 0.5, // 0 -> A (train/test assignment as below), 1 -> B (train/test assignment inverted)

        // Path to save the data
        path_data: 'tasks/kai/HumanObserveBetEfficacyDay1Of2/data/',
        temp_path_data: 'tasks/kai/HumanObserveBetEfficacyDay1Of2/data/temp/',
        resize_path_data: 'tasks/kai/HumanObserveBetEfficacyDay1Of2/data/resize/',

        // EXPERIMENTAL PARAMETERS
        n_arms: 2,
        redraw_vol: 0.20, 
        bias: 0.5,

        t_feedback: 1000, // in milliseconds, originally 1500

        // INVERTED SETTINGS - CORRESPONDING TO GROUP A FOR THE MAIN EXPERIMENT
        effs_train : [{'eff': 1}, {'eff': 0}, {'eff': 0.25}, {'eff': 0.75}],
        effs_test : [{'eff': 0.125}, {'eff': 0.375}, {'eff': 0.5}, {'eff': 0.625}, {'eff': 0.875}],
        n_episodes_train : 4,
        n_episodes_test : 5,

        show_quiz : true,
        show_instructions : true,
        n_steps: 50,

    }

    // FLIP TRAINING AND TEST ASSIGNMENTS IF IN GROUP B

    if (params.group) {
        let temp_effs_train = params.effs_train
        let temp_effs_test = params.effs_test
        params.effs_train = temp_effs_test
        params.effs_test = temp_effs_train

        let temp_n_episodes_train = params.n_episodes_train
        let temp_n_episodes_test = params.n_episodes_test
        params.n_episodes_train = temp_n_episodes_test
        params.n_episodes_test = temp_n_episodes_train

    }

    return params
}