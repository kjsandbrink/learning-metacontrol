function trialMain(params) {
    
    // initializations
	var i_step = 1;
	var i_episode = 0;
	var eff;
	var efficacy_estimate;
	var rewards_tally = 0; //per-episode
	var total_rewards_tally = 0; //total experiment
	var ps = [0, 0];
	var transitions_ep = [];
	var ps_ep = [];

	// GAMEPLAY FUNCTIONS

	function trial_number_html() {
		return `<p>Round ` + i_step + `/` + params.n_steps + `</p>`;
	}

	function reset_ps(){
		var bias_dir = (Math.random()>=0.5)? 1 : -1;
		ps = [0.5 + bias_dir*params.bias, 0.5 - bias_dir * params.bias];
	}

	function setup(){
		//console.log("starting setup")
		rewards_tally = 0;
		reset_ps();
		//console.log(ps);
		transitions_ep = [];
		i_step = 1;
		ps_ep = [];

	}
	setup();

	// CREATE TIMELINE


	var timeline = [];

	var iti = {
		type: jsPsychHtmlButtonResponse,
		
		// FORCE TO WAIT
		choices: [],
		stimulus: `<p>&nbsp;</p><p>&nbsp;</p><p>&nbsp;</p>` + format_image_button("lights/Slide1") +
		`<div>` + format_image_button("observe/Slide1") + `</div>` + format_image_button("lights/Slide2") + `<p>&nbsp;</p>`,

		trial_duration: 350 // 0.5 milliseconds		  
	}

	// MAIN EXPERIMENT
	//timeline.push(iti)

	var step = {
		type: jsPsychHtmlButtonResponse,
		choices: ["lights/Slide1", "observe/Slide1", "lights/Slide2"],

		stimulus: function(data) {
			return trial_number_html() + `<p>Bet: Click on the light of your choice</p><p>Observe: Click on the glasses</p>`;
		},

		button_html: function(){
		
			var html = format_image_button("%choice%");
			
			return html;
			
		},

        margin_horizontal: '400px',

		prompt: `<p>Click on one of the pictures to continue</p>`,

		on_finish: function(data) {
			// PROCESS RESPONSE

			//recalibrate choice - this is the variable with which we proceed calculation
			// 0-1: arm was taken
			// 2: observe
			if (data.response == 0) {
				data.choice = data.response
			}
			else {
				data.choice = 3 - data.response
			}

			// determine which arm was rewarded
			data.rewarded_arm = (Math.random()<ps[0])? 0 : 1;
			eff = jsPsych.timelineVariable('eff')
			data.eff = eff

			// if in "take" mode determine if choice was successful and save chosen arm
			//i.e. bet option
			if (data.choice <= 1) {

				// EFFICACY IMPLEMENTATION
				if (Math.random() > eff) { 
					//efficacy gives 1 - the chance of a random arm being drawn
					data.taken_arm = (Math.random()<0.5)? 0 : 1;
				}
				else {
					data.taken_arm = data.choice
				}

				data.reward = (data.taken_arm == data.rewarded_arm)? 1 : 0;

				rewards_tally += data.reward;
				total_rewards_tally += data.reward;

			}
			else {
				data.reward = 0
				data.taken_arm = 2
			}

			// VOLATILITY IMPLEMENTATION
			data.ps = ps
			ps_ep.push(ps)
			if (Math.random() < params.redraw_vol) {
				reset_ps()
			}

			transitions_ep.push([data.rewarded_arm, data.choice, data.taken_arm])
		}
	};

	var feedback = {
		type: jsPsychHtmlButtonResponse,

		// FORCE TO WAIT
		choices: [],
		prompt:  `<p>&nbsp;</p>`,
		stimulus: function(data) {

			var choice = transitions_ep[transitions_ep.length - 1][1];
			var rewarded_arm = transitions_ep[transitions_ep.length - 1][0];
			//console.log(choice)

			// set feedback for BET trials
			if (choice <= 1){
				var taken_arm = transitions_ep[transitions_ep.length - 1][2];
				// for text to be written on top
				chosen_color = chooseColor(choice);
				taken_color = chooseColor(taken_arm);

				// prepare action
				feedback_chosen_arm = `<span  style='color:` + chosen_color.toLowerCase() + `;'>You Wanted ` + chosen_color + `</span>`;
				feedback_taken_arm = `<span style='color:` + taken_color.toLowerCase() + `;'>You Bet ` + taken_color + `</span>`;
				feedback_text = `<p>` + feedback_chosen_arm + `<span style="display:inline-block; width: 100px;"></span>` + feedback_taken_arm + `</p>`; 
				feedback_instructions = `<p>If ` + chosen_color.toLowerCase() + ` was the correct light, you will have earned a coin</p>`
			}

			// for OBSERVE trials
			else {
				color = chooseColor(rewarded_arm);
				feedback_text =  `<p style='color:`+ color.toLowerCase() + `;'>` + color + ` Lit Up</p>`;
				feedback_instructions = `<p>If you had bet on ` + color.toLowerCase() + `, you would have earned a coin</p>`
			}

			// compile feedback
			feedback = trial_number_html() + feedback_instructions + feedback_text;

			// set feedback for BET trials
			if (choice <= 1){
				var taken_arm = transitions_ep[transitions_ep.length - 1][2];

				if(taken_arm == 0 && choice == 0) {
					feedback_blue = format_image_button('lights/Slide9');
					feedback_red = format_image_button('lights/Slide2');
				}
				else if(taken_arm == 0 && choice == 1) {
					feedback_blue = format_image_button('lights/Slide7');
					feedback_red = format_image_button('lights/Slide4');
				}
				else if(taken_arm == 1 && choice == 0) {
					feedback_blue = format_image_button('lights/Slide3');
					feedback_red = format_image_button('lights/Slide8');
				}
				else {
					feedback_blue = format_image_button('lights/Slide1');
					feedback_red = format_image_button('lights/Slide10');
				}

				feedback_observe = format_image_button('observe/Slide1')				
				
				// for text to be written on top
				chosen_color = chooseColor(choice);
				taken_color = chooseColor(taken_arm);
			}

			// for OBSERVE trials
			else {

				if (rewarded_arm == 0) {
					feedback_blue = format_image_button('lights/Slide11')
					feedback_red =  format_image_button('lights/Slide2')		
				}
				else {
					feedback_blue = format_image_button('lights/Slide1')
					feedback_red =  format_image_button('lights/Slide12')	
				}

				// OBSERVE SIGNAL
				// without -1
				feedback_observe = format_image_button('observe/Slide2')
				// with +1

				color = chooseColor(rewarded_arm);
				feedback_text =  `<p style='color:`+ color.toLowerCase() + `;'>You Observed ` + color + `</p>`;
				feedback_instructions = `<p style='color:`+ color.toLowerCase() + `;'>If you had bet on ` + color.toLowerCase() + `, you would have earned a coin</p>`
			}

			// compile feedback
			feedback += `<div>` + feedback_blue + `</div><div>` + feedback_observe + `</div><div>` + feedback_red + `</div>`;

			return feedback;
		},

		on_finish : function() {
			// increment next trial
			i_step ++ ;
		},

		trial_duration: params.t_feedback,

        margin_horizontal: '400px',
	};

	var trial = {
		timeline: [iti, step, feedback],
		randomize_order: false,
		repetitions: params.n_steps,
	};

	// BUILD CUE FOR LEARNING

	var efficacy_cue = {
		type: jsPsychHtmlButtonResponse,
		choices : ['Continue'],
		stimulus : function() {
			eff = jsPsych.timelineVariable('eff')
			console.log('efficacy', eff)
			if (eff == 0) {
				descriptor_control = 'no';
				descriptor_machine = 'half the time (entirely unpredictably, so it is impossible to time your bets accordingly)';
			}
			else if (eff < 1/3) {
				descriptor_control = 'a little';
				descriptor_machine = 'slightly more than half the time (entirely unpredictably, so it is impossible to time your bets accordingly)';
			}
			else if (eff < 2/3) {
				descriptor_control = 'some';
				descriptor_machine = 'most';
			}
			else if (eff < 1) {
				descriptor_control = 'a lot of';
				descriptor_machine = 'almost always';
			}
			else {
				descriptor_control = 'complete';
				descriptor_machine = 'always';
			}
			
			return `<p>In this set, you will have <b>` + descriptor_control + ` control,</b> meaning that you will <b>` + descriptor_machine + ` </b> be able to successfully place your bet on the light you intend.</p>`+ 
				`<p>This is indicated by the background color, and will stay the same until at least the end of this set. ` +
				`In the final ` + params.n_episodes_test + ` sets, the background color will change to match your level of control, but you will no longer see this explanatory screen.</p>` +
				`<p><b><i>Note that observing unnecessarily or in situations where you have no control will reduce your reward due to opportunity costs.</i></b></p>`
		},
		on_start : set_efficacy_cue_style,
		trial_duration : 0,
	}

	var efficacy_survey = {
		type: jsPsychSurveyText,
		questions: [
		  {prompt: 'How likely do you think you were to be able to place your bet successfully?', required: true},
		  {prompt: 'In case you didn\'t have any control, did you still spend time observing? How come?', required: false, rows: 3},
		  {prompt: 'If so, what would need to have been changed for you not to have spent any time observing?', required: false, rows: 3}
		],

		on_start: reset_style,

	  }

	var efficacy_slider = {
		type: jsPsychHtmlSliderResponse,
		stimulus: `<p>How likely do you think you were able to successfully place your bets on the light that you intended in this set? How much control do you think you had? [in \%]</p>`,
		require_movement: true,
		min : 50,
		max : 100,
		slider_start: 75,
		on_start : reset_style,
		slider_width: 400,
		labels : ['50%', '62.5%', '75%', '87.5%', '100%'],
		on_finish (data) {
			efficacy_estimate = data.response;
			console.log(efficacy_estimate);
		}
	}


	var save = {
		type: jsPsychHtmlButtonResponse,
		stimulus: '',
		choices : [],
		trial_duration : 0,
		on_finish : function () { 				
				saveData(params, jsPsych.data.getLastTimelineData().filter({end_ep: true}).last(1).csv(), {i_episode:i_episode, temporary : false}) ;
			},

		on_start: reset_style,
	}

	
	// TRAIN TRIALS

	var announce_training = {
		type : jsPsychHtmlButtonResponse,
		choices: ["Continue"],
		stimulus: "<p>We will now begin with the sets!</p>" +
			"<p>For these first " + params.n_episodes_train + " sets, you will see the total sequence of which lights lit up and which actions you chose at the end of the 50 steps. For the final sets, you will only see the total amount of points that you earned.</p>" + 
			"<p>The game begins on the next page!</p>",
	};
	timeline.push(announce_training);

    function build_trajectory_feedback() {
        row_titles = ['Correct', 'Chosen', 'Taken'];

        header = `<p>Set ` + (i_episode + 1).toString() + `/` + params.n_episodes_train + `</p>`;

        image_grid = `<table style="padding-right: 7px; margin-left:auto; margin-right:auto;">`;

        for (let i = 0; i < 2; i++) {

            for (let k = 0; k < 3; k++) {
                image_grid += `<tr><td>` + row_titles[k] + `</td>`;

                for (let j = 0; j < params.n_steps / 2; j++) {
                    image_grid += `<td><img src='stim/feedback_elements/Slide` + (1 + transitions_ep[ i*(params.n_steps/2) + j ][k]) + `.JPG' style='max-height:25px;'></img></td>`;
                }

                image_grid += `</tr>`
            }

            image_grid += '<tr><td></td>'

            for (let j = 0; j < params.n_steps / 2; j++) {
                image_grid += `<td><p>` +( i*(params.n_steps/2) + j + 1 )+ `</p></td>`
            }

            image_grid +='</tr>'

        }

        image_grid += `</table>`

        return header + image_grid;
    }

    var feedback_train_episode = {
        type: jsPsychInstructions,
        pages : function() {

            pages = [`<p>On the next screen, you will first be shown a <strong>summary of how many points you earned</strong>, and then on the following one, an <strong>overview of the trajectory</strong> showing you which arm was the right one on a given trial, ` +
                    `whether you chose to observe or bet on a light, and, if you bet, which arm you chose to bet on and which arm you chose to actually bet on.</p>`,
            `<p>Set ` + (i_episode + 1).toString() + `/` + params.n_episodes_train + `</p><p>Rewards earned this episode: ` + rewards_tally + `</p><p>Total rewards earned: ` + total_rewards_tally + `</p>`,
            build_trajectory_feedback(),
            ];

            if (i_episode + 1 < params.n_episodes_train) {
                this.pages.push(`<p>The next set will begin on the next page.</p>`)
            }

            return this.pages;
        } ,

        on_finish : function end_ep (data) {
			data.rewards_tally = rewards_tally;
			data.total_rewards_tally = total_rewards_tally;
			data.transitions_ep = transitions_ep
			data.ps_ep = ps_ep;
			data.id = params.id;
			data.eff_ep = eff;
			data.sex = participant_gender;
			data.efficacy_estimate = efficacy_estimate;
			data.age = participant_age;
			data.turker = participant_turker;
			data.end_ep = true;
			data.group = params.group;
			i_episode++;
			setup();
		},

        show_clickable_nav : true

	};

	var train_episode = {
		timeline: [efficacy_cue, trial, efficacy_slider, feedback_train_episode, save],
		timeline_variables: params.effs_train,
		randomize_order: true,
		sample: {
			type: 'without-replacement',
		},	

	};

	timeline.push(train_episode);

	// TEST EPISODES

	var announce_testing = {
		type : jsPsychHtmlButtonResponse,
		choices: ["Continue"],
		stimulus: "<p>We will now begin with the final group of sets!</p>" +
			"<p>For these final " + params.n_episodes_test + " sets, you will only see your chosen reward at the end of a set of 50 rounds, and no longer the individual sequence of actions and lights.</p>" + 
			"<p>The game begins on the next page!</p>",
	};
	timeline.push(announce_testing);

    var feedback_test_episode = {
        type: jsPsychInstructions,
        pages : function() {

            pages = [`<p>Congratulations on finishing the set!</p><p>On the next screen, you will see a <strong>summary of how many points you earned</strong>.`,
            `<p>Set ` + (i_episode - params.n_episodes_train + 1).toString() + `/` + params.n_episodes_test + `</p><p>Rewards earned this episode: ` + rewards_tally + `</p><p>Total rewards earned: ` + total_rewards_tally + `</p>`,
            ];

            if (i_episode + 1 < params.n_episodes_test) {
                this.pages.push(`<p>The next set will begin on the next page.</p>`)
            }

            return this.pages;
        } ,

        on_finish : function end_ep (data) {
			data.rewards_tally = rewards_tally;
			data.total_rewards_tally = total_rewards_tally;
			data.transitions_ep = transitions_ep
			data.ps_ep = ps_ep;
			data.id = params.id;
			data.eff_ep = eff;
			data.sex = participant_gender;
			data.efficacy_estimate = efficacy_estimate;
			data.age = participant_age;
			data.turker = participant_turker;
			data.end_ep = true;
			data.group = params.group;
			i_episode++;
			setup();
		},

        show_clickable_nav : true

	};

	var test_episode = {
		timeline: [trial, efficacy_slider, feedback_test_episode, save],
		timeline_variables: params.effs_test,
		randomize_order: true,
		sample: {
			type: 'without-replacement',
		},
	};

	timeline.push(test_episode);

    return timeline;

}

function blocMain(params) {
    
    let bloc = {
        timeline: trialMain(params),
    }
    return bloc
}
