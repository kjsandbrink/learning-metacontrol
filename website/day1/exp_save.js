function get_date() {
    var date = new Date();
    return (date.getFullYear() + ("0" + (date.getMonth() + 1)).slice(-2) + ("0" + date.getDate()).slice(-2) + ("0" + date.getHours() ).slice(-2) + ("0" + date.getMinutes()).slice(-2) + ("0" + date.getSeconds()).slice(-2)).toString();
    
}

function saveData(params, data, {i_episode = -1, temporary = true, resize = false, suffix = ''}) {
    //console.log(i_episode)
    let suffix_ep;
    if (i_episode != -1) {
        suffix_ep = '_ep' + i_episode.toString();
    }
    else {
        suffix_ep = '';
    }

    //console.log(suffix_ep)
    
    
    // Set the data to be saved
    const alldata = {
        //id: params.id, //TODO: Change here to test if this determines name that file is saved under
        id: get_date() + '_' + participant_turker + '_' + params.id + suffix_ep + suffix,
        task: params.task,
        path: function() {
            if (resize) {
                return params.resize_path_data
            } else {
                if (temporary) {
                    return params.temp_path_data
                } else {
                    return params.path_data
                }
            }
        },
        data: data,
    }

    // Send it to the back-end (Perl)
    logWrite(alldata)
}
