function returnExperimentParameters() {
    let params = {

        // General parameters
        task: task_name,
        id: jsPsych.randomization.randomID(10),
        group: true,

        // Path to save the data
        path_data: 'tasks/kai/HumanObserveBetEfficacyDay3Of3B/data/',
        temp_path_data: 'tasks/kai/HumanObserveBetEfficacyDay3Of3B/data/temp/',
        resize_path_data: 'tasks/kai/HumanObserveBetEfficacyDay3Of3B/data/resize/',

        // EXPERIMENTAL PARAMETERS
        n_arms: 2,
        redraw_vol: 0.10, 
        bias: 0.5,
        eff_bonus_strength: 0.1,
        eff_limit: 1,

        t_feedback: 1000, // in milliseconds, originally 1500

        effs_train : [{'eff': 1}, {'eff': 0}, {'eff': 0.25}, {'eff': 0.75}],
        effs_test : [{'eff': 0.125}, {'eff': 0.375}, {'eff': 0.5}, {'eff': 0.625}, {'eff': 0.875}],
        n_episodes_train : 4,
        n_episodes_test : 5,

        show_quiz : true,
        show_instructions : true,
        n_steps: 50,
    }
    return params
}