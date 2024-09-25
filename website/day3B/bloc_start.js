function blocStart(params) {
    var preload = {
        type: jsPsychPreload,
        images: function () {
            images_to_preload = [];
            for (let i = 1; i <= 12; i++) {
                images_to_preload.push(['stim/lightstask2/Slide' + i + '.JPG']);
            }
            for (let i = 1; i <= 3; i++) {
                images_to_preload.push(['stim/observetask2/Slide' + i + '.JPG']);
            
            }
            for (let i = 1; i <= 4; i++) {
                images_to_preload.push(['stim/feedback_elements/Slide' + i + '.JPG']);
            
            }
            for (let i = 1; i <= 1; i++) {
                images_to_preload.push(['stim/instructionstask2/Slide' + i + '.JPG']);
            
            }
			for (let i = 1; i <= 2; i++) {
                images_to_preload.push(['stim/sleeptask2/Slide' + i + '.JPG']);
            
            }
			for (let i = 1; i <= 2; i++) {
                images_to_preload.push(['stim/sleep/Slide' + i + '.JPG']);
            
            }
            images_to_preload.push(['stim/background/casino.png'])
            return images_to_preload;
        }
    }

    var instructions_timeline = [preload];

    if (params.show_instructions) {
        var instructions_recap = {
			type : jsPsychInstructions,
			pages : [
					'<h1>Casino Game Day 3</h1><p>Welcome</p>',
					`<p style='text-align:left'>On days 1 and 2, you had the option of <b>observing</b> or <b>betting</b>, and depending on which option you chose, would either receive feedback about which light was the 'lucky' one or had the chance to earn reward that would be added to a secret tally.</p>`,
					`<img style='max-width: 150px;' src='stim/sleep/Slide1.JPG'></src><img style='max-width: 150px;' src='stim/sleep/Slide2.JPG'></src><p style='text-align:left'>Today, you will additionally have the option of selecting to <b>sleep</b> by clicking on the picture of a bed. You won't have the chance to earn any additional reward when you sleep, but doing so will increase your chance placing your intended bets successfully for the remainder of the set.</p><p style='text-align:left'>Sleeping takes the same amount of time as placing a bet or observing.</p>`,
					`<p style='text-align:left'>Like previously in Day 2, you will see the same background colors for the first ` + params.n_episodes_train + ` sets that indicate your current level of efficacy, but without specific instructions, followed by ` + params.n_episodes_test + ` sets without any indication at all but where you will need to infer the amount of control based on your success in placing your bets.</p>`,
					`<p style='text-align:left'>Because you have more gains to make by sleeping in low-control settings, it makes sense to <b>sleep more when you have less control and less in high-control settings</b> where there is less room to increase control.</p>` +
					`<p style='text-align:left'>In general, the lucky light switches less frequently this round, <b>so it is not necessary to observe as often.</b></p>`,
					`<p style='text-align:left'>On the next pages, you will see a quick summary of the rules followed by a multiple choice quiz on strategy (review from day 1). After that, the game will start. It won't be possible to return to these pages after you click next, so if you want to read the instructions one more time, please do so now.</p>`
					
			],
			show_clickable_nav : true,

			on_start: set_html_style_instructions,
			on_finish: set_html_style_normal,

		}

		instructions_timeline.push(instructions_recap)
		
		var announce_survey = {
			type : jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus:`<p>After the end of the game, we will ask you to fill out some post-questionnaires (multiple choice, ~15 minutes), <b>so please do not exit the game until you see the final completion code!</b></p>`,
		};
		instructions_timeline.push(announce_survey);

		var teach_sleep_new_lights = {
			type: jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus: `<p>Here is an explanation of the new task:</p>`,
			button_html: `<img style="max-width:650px;" src="stim/instructionstask2/Slide1.JPG">`,
			prompt: `<p>Click on the pictures to continue</p>`,
			margin_vertical: '18px',
		};
		instructions_timeline.push(teach_sleep_new_lights)

		var instructions_plain = {
			type : jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus: `<p>Here is a quick recap of the overall rules for today:</p>` +
			`<ul style='text-align:left'>` +
			`<li>Two lights - one red, one blue - one is the 'lucky' one</li>` +
			`<li>The 'lucky' light can switch at any time</li>` +
			`<li>Every round, you choose between observing, betting, or sleeping</li>` + 
			`<li>If you sleep, you will increase your odds of successfully placing a bet on the light you intend for the remainder of the set</li>` +
			`<li>The lucky light switches less frequently in this round so in general it is not necessary to observe as often</li>` +
			`<li>You will play ` + (params.n_episodes_test + params.n_episodes_train).toString() + ` sets in total</li>` +
			`<li>For the first ` + (params.n_episodes_train).toString() + ` sets, you will receive a background color cue indicating your current level of efficacy</li>` +
			`<li>For the final ` + (params.n_episodes_test).toString() + ` sets,  without any indication at all but where you will need to infer the amount of control based on your success in placing your bets</li>` +
			`</ul>`,
		};
		instructions_timeline.push(instructions_plain);
    }

    
	if(params.show_quiz) {

		quiz_uncertainty_prompts = [
		"What is the effect of sleeping?",
		"In which cases is it worth sleeping more? When is it worth sleeping less or not at all?", 
		"What does the background color cue that is available in the first few sets indicate?",
		"What is a good straegy for choosing at which times to observe? How does this change based on your degree of control?",
			]

		quiz_uncertainty_options = [
						[
							"Sleeping has no effect",
							"Sleeping increases your probability of placing bets successfully for the next turn only",
							"Sleeping increases your probability of placing bets successfully for the rest of the set",
							"Sleeping increases your probability of placing bets successfully for the rest of the game",
						],[
							"It is always worth sleeping",
							"It is worth sleeping more in cases where you have less control, and worth sleeping less or none at all in cases where you have more control",
                            "It is worth sleeping more in cases where you have more control, and worth sleeping less or none at all in cases where you have less control",
                            "It is never worth sleeping",
						],[
							"The background cue always indicates your level of control at the start of the set",
							"The background cue always indicates your current level of control, taking into account changes coming from sleeping",
							"The background cue does not hold any information about your level of control",	
						],
						[
							"Observing a lot at the beginning and then not anymore",
							"Observing every once in a while, as observing will tell me the lucky light for a certain period, but the lucky light can switch after that",
							"Observing every once in a while, as observing will tell me the lucky light for a certain period, but the lucky light can switch after that",
                            "It is impossible to know what the lucky light is so I can just guess randomly",
						],
		]

		quiz_uncertainty_idxs_correct = [2, 1, 1, 2, ]

		solutions_uncertainty_pictures = ["",
			"<img style='max-width: 75px;' src='stim/lights/Slide3.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide4.JPG'></src>",
			"<img style='max-width: 75px;' src='stim/lights/Slide7.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide8.JPG'></src>"
		]

		instructions_timeline = instructions_timeline.concat(create_quiz(quiz_uncertainty_prompts, quiz_uncertainty_options, quiz_uncertainty_idxs_correct, solutions_uncertainty_pictures))

	}

    var bloc = {
        timeline : instructions_timeline,
    }

    return bloc
}