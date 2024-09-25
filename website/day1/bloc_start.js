function blocStart(params) {
    var preload = {
        type: jsPsychPreload,
        images: function () {
            images_to_preload = [];
            for (let i = 1; i <= 12; i++) {
                images_to_preload.push(['stim/lights/Slide' + i + '.JPG']);
            }
            for (let i = 1; i <= 3; i++) {
                images_to_preload.push(['stim/observe/Slide' + i + '.JPG']);
            
            }
            for (let i = 1; i <= 3; i++) {
                images_to_preload.push(['stim/feedback_elements/Slide' + i + '.JPG']);
            
            }
            for (let i = 1; i <= 7; i++) {
                images_to_preload.push(['stim/instructions/Slide' + i + '.JPG']);
            
            }
            images_to_preload.push(['stim/background/casino.png'])
			images_to_preload.push(['stim/slider.png'])
            return images_to_preload;
        }
    }

    var instructions_timeline = [preload];

    if (params.show_instructions) {
        var instructions = {

            type: jsPsychInstructions,

				// REWRITE 5/11 - CONCISION AND CLARITY

				pages : [ `<h1>Casino Game</h1><p>Welcome</p>`,
                `<p style='text-align:left'>You enter an unusual casino and approach one of the tables which contains a game you have not seen before, featuring two lights in the middle of the table. You ask the employee behind the table, "How do I play?"</p>`,
                `<img style='max-width: 150px;' src='stim/lights/Slide1.JPG'></src>&nbsp;&nbsp;<img style='max-width: 150px;' src='stim/lights/Slide2.JPG'></src><p style='text-align:left'>"Simple," she says. "One of these two lights is <b>the 'lucky' light</b> and pays out coins if you bet on it.  The 'lucky' light will switch occasionally (there's roughly a 10\% chance of it changing after every round, so on average it will switch once every 10 rounds). Each round, you can pick whether you want to bet for money, or to observe.</p>`,
                `<img style='max-width: 150px;' src='stim/observe/Slide1.JPG'></src><p style='text-align:left'>"If you <b>observe</b>, the 'lucky' light lights up - that is the light that would have paid out if you had chosen it.</p><img style='max-width: 150px;' src='stim/lights/Slide5.JPG'></src>&nbsp;&nbsp;<img style='max-width: 150px;' src='stim/lights/Slide6.JPG'>`,
				`<img style='max-width: 150px;' src='stim/lights/Slide9.JPG'></src>&nbsp;&nbsp;<img style='max-width: 150px;' src='stim/lights/Slide10.JPG'></src><p style='text-align:left'>"If you <b>bet</b>, you'll choose one of the lights. If you guessed correctly, you will earn a coin. However, the machine won't tell you whether you've won or not - but it will add a coin to your prize money when you do.</p><p style='text-align:left'>"After 50 rounds, I'll reveal the total number of coins you've amassed. Want to play?"</p>`,
				`<p style='text-align:left'>At your nod of agreement, she continues, "There's one more thing to note: <b>sometimes, you may choose to bet (and click on) one light, but you will accidentally place your bet on the other light instead.</b> The chance of this happening remains the same within a set, and will be high in some and low in others. You will need to find out by playing and seeing if you execute actions successfully how much control you have in each set. <b><i>However, it will never be higher than 50\% (and occur randomly without patterns), so it is never beneficial to intentionally bet on the opposite light.</b></i></p><img style='max-width: 150px;' src='stim/lights/Slide3.JPG'></src><img style='max-width: 150px;' src='stim/lights/Slide8.JPG'></src>&nbsp;&nbsp;<img style='max-width: 150px;' src='stim/lights/Slide7.JPG'></src><img style='max-width: 150px;' src='stim/lights/Slide4.JPG'></src>`,// Instead, in general, it is a good strategy to observe more in rounds where you have more control and less in rounds where you have less control.</p>
                `<img style='max-width: 425px;' src='stim/slider.png'></src><p style='text-align:left'>"After a set, you will see a slider that you will need to use to <b>estimate the amount of control you had</b> (i.e. how likely you were to be able to place your bets successfully) in the previous set.</p>`,
				`<p style='text-align:left'>"All in all, you will play `+ (params.n_episodes_test + params.n_episodes_train).toString() + ` sets of 50 rounds. After each set, you will see a detailed summary of your actions and the lights that lit up on every round. Are you ready to give it a try?"</i></b>`,
                `<p style='text-align:left'>On the next pages, you will see the rules again in condensed form, play practice rounds, and answer questions about them. It won't be possible to return to these pages after you click next, so if you want to read the story one more time, please do so now.</p>`
                ],

            
                show_clickable_nav : true,

                on_start: function() {
                    saveData(params, jsPsych.data.get().csv(), {temporary:true});
                    set_html_style_instructions();
                },
                on_finish: set_html_style_normal

        }

        instructions_timeline.push(instructions)
            
    }

    
	if(params.show_quiz) {

		// REMAINING INSTRUCTION TRIALS
		var instructions_observe = {
			type : jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			//stimulus: "<p>Let's try observing!</p>"
			stimulus: "<p>On the next screens, we will take a look at what options will look like during the game!</p>" +
				"<p>We will walk through each possible action and take a look at the corresponding feedback screens, beginning with observe actions and then looking at bets that are switched and those that aren't.</p>"
		};
		instructions_timeline.push(instructions_observe);

		var teach_observe = {
			type: jsPsychHtmlButtonResponse,
			choices: ["observe/Slide1"],

			stimulus: function(data) {
				return `<p>Practice</p><p>&nbsp;</p><p>Observe: Click on the glasses</p>` + format_image_button("lights/Slide1");
			},

			button_html: function(){
			
				var html = format_image_button("%choice%");
				
				return html;
				
			},

			margin_horizontal: '400px',

			prompt: format_image_button("lights/Slide2") + `<p>Click on one of the pictures to continue</p>`,
		};

		//timeline.push(iti)
		instructions_timeline.push(teach_observe)

		var teach_observe_feedback = {
			type: jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus: `<p>Great!</p><p>Here is an explanation of feedback you might get after observing:</p>`,
			button_html: `<img style="max-width:650px;" src="stim/instructions/Slide2.JPG">`,
			prompt: `<p>Click on the pictures to continue</p>`,
			margin_vertical: '18px',
		};

		instructions_timeline.push(teach_observe_feedback)

		quiz_observe_prompts = [ "Which button did you just press in order to observe?", 
			"What does the appearance of a darkened circle represent after you observed? <img style='max-width: 75px;' src='stim/lights/Slide11.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide12.JPG'></src>",
			"What does the appearance of a light circle after you observed represent? <img style='max-width: 75px;' src='stim/lights/Slide1.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide2.JPG'></src>"
			]

		quiz_observe_options = [
						[
							"<img style='max-width: 75px;' src='stim/observe/Slide1.JPG'></src>",
							"<img style='max-width: 75px;' src='stim/lights/Slide1.JPG'></src>",
							"<img style='max-width: 75px;' src='stim/lights/Slide2.JPG'></src>"
						],
						[
							"It signifies that you've earned a coin as the corresponding light lit up",
							"The corresponding light lit up, indicating that it's the 'lucky' light and you would have won a coin if you had bet on it",
							"You successfully placed a bet on the corresponding color",
							"You attempted to bet on the corresponding color, but your bet switched to the other color"
						] ,
						[
							"It means the corresponding light lit up and you've earned a coin",
							"You attempted to bet on this light color, but your bet was inadvertently switched to the other light",
							"You successfully placed a bet on this light color", 
							"The corresponding light didn't light up, indicating that it's not the 'lucky' light and you wouldn't have won a coin if you had bet on it",
						]
		]

		quiz_observe_idxs_correct = [0, 1, 3]

		solutions_observe_pictures = ["",
			"<img style='max-width: 75px;' src='stim/lights/Slide11.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide12.JPG'></src>",
			"<img style='max-width: 75px;' src='stim/lights/Slide1.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide2.JPG'></src>"
		]

		instructions_timeline = instructions_timeline.concat(create_quiz(quiz_observe_prompts, quiz_observe_options, quiz_observe_idxs_correct, solutions_observe_pictures))
		
		var instructions_bet_successful = {
			type : jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus: "<p>Now, let's take a look at bet actions where you manage to place the bet on the arm you intended.</p>",
		};
		instructions_timeline.push(instructions_bet_successful);

		var teach_bet_successful_blue = {
			type: jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			//prompt: "<p>Which mode and arm do you choose?</p>"

			stimulus: function(data) {
				return `<p>Practice Round</p><p>Successful Bet on Blue</p><p>&nbsp;</p>`;
			},

			button_html: function() {
				return format_image_button("lights/Slide1");
			},

			prompt: function() {
				return `<div>` + format_image_button("observe/Slide1") + `</div>` + format_image_button("lights/Slide2") + `<p>Click on the blue arm</p>`;
			},

			//margin_vertical: '18px',
		};
		//timeline.push(iti)
		instructions_timeline.push(teach_bet_successful_blue)

		// HERE
		var feedback_bet_successful_blue = {
			type: jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus: `<p>Great! Here is an explanation of feedback you might get after successfully placing a bet:</p>`,
			button_html: `<img style="max-width:675px;" src="stim/instructions/Slide4.JPG">`,
			prompt: `<p>Click on the pictures to continue</p>`,
			margin_vertical: '18px',
		};

		instructions_timeline.push(feedback_bet_successful_blue)

		var teach_bet_successful_red = {
			type: jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			//prompt: "<p>Which mode and arm do you choose?</p>"

			stimulus: function(data) {
				return `<p>Practice Round</p><p>Successful Bet on Red</p><p>&nbsp;</p>` + format_image_button("lights/Slide1") +
				`<div>` + format_image_button("observe/Slide1") + `</div>`;
			},

			button_html: function () {
				return format_image_button("lights/Slide2");
			},
			//margin_vertical: '18px',

			prompt: `<p>Click on one of the pictures to continue</p>`,

		};
		//timeline.push(iti)
		instructions_timeline.push(teach_bet_successful_red)

		var feedback_bet_successful_red = {
			type: jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus: `<p>Great!<p></p>Here is an explanation of feedback you might get after successfully placing a bet:</p>`,
			button_html: `<img style="max-width:675px;" src="stim/instructions/Slide6.JPG">`,
			prompt: `<p>Click on the pictures to continue</p>`,
			margin_vertical: '18px',
		};

		instructions_timeline.push(feedback_bet_successful_red)

		quiz_prompts = [ "In order to place a bet, you clicked on one of which pair of buttons in the two preceding rounds?", 
			"What does the appearance of a circle with a ring, coin, and question mark after you bet on a light represent? <img style='max-width: 75px;' src='stim/lights/Slide9.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide10.JPG'></src>",
			"What does the appearance of a light circle after you placed a bet represent? <img style='max-width: 75px;' src='stim/lights/Slide1.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide2.JPG'></src>"
			]

		quiz_options = [
						[
							"<img style='max-width: 75px;' src='stim/lights/Slide1.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide2.JPG'></src>",
							"<img style='max-width: 75px;' src='stim/observe/Slide1.JPG'></src><img style='max-width: 75px;' src='stim/observe/Slide2.JPG'></src>",
						],
						[
							"The corresponding light lit up, indicating that you've won a coin",
							"The corresponding light lit up, signifying that you could have won a coin if you had bet on it, but you didn't win anything this round",
							"You successfully placed a bet on the corresponding color, implying potential winnings if the corresponding light lit up (not known at the moment)",
							"You attempted to bet on the corresponding color, but your bet switched to the other color, implying potential winnings if the other light lit up (not known at the moment)"
						] ,
						[
							"It means the corresponding light lit up and you've earned a coin",
							"You did not place a bet on this color",
							"You successfully placed a bet on this color",
							"The corresponding light didn't light up, indicating that you wouldn't have won a coin if you had bet on it"
						]
		]

		quiz_idxs_correct = [0, 2, 1]

		solutions_pictures = ["",
			"<img style='max-width: 75px;' src='stim/lights/Slide9.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide10.JPG'></src>",
			"<img style='max-width: 75px;' src='stim/lights/Slide1.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide2.JPG'></src>"
		]

		instructions_timeline = instructions_timeline.concat(create_quiz(quiz_prompts, quiz_options, quiz_idxs_correct, solutions_pictures))

		var instructions_bet_unsuccessful = {
			type : jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus: "<p>Great! Now, let's try a round where you try to bet on a light of one color <br> but accidentally choose the other light instead.</p>"
		};
		instructions_timeline.push(instructions_bet_unsuccessful);

		var teach_bet_unsuccessful_blue = {
			type: jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			//prompt: "<p>Which mode and arm do you choose?</p>"

			stimulus: function(data) {
				return `<p>Practice Round</p><p>Bet on Blue That Is Switched</p><p>&nbsp;</p>`;
			},

			button_html: function() {
				return format_image_button("lights/Slide1");
			},

			prompt: function() {
				return `<div>` + format_image_button("observe/Slide1") + `</div>` + format_image_button("lights/Slide2") + `<p>Click on the blue arm</p>`;
			},

			//margin_vertical: '18px',
		};
		//timeline.push(iti)
		instructions_timeline.push(teach_bet_unsuccessful_blue)

		var feedback_bet_unsuccessful_blue = {
			type: jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus: `<p>Correct!</p>`,
			button_html: `<img style="max-width:675px;" src="stim/instructions/Slide5.JPG">`,
			prompt: `<p>Click on the pictures to continue</p>`,
		};

		instructions_timeline.push(feedback_bet_unsuccessful_blue)

		var teach_bet_unsuccessful_red = {
			type: jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			//prompt: "<p>Which mode and arm do you choose?</p>"

			stimulus: function(data) {
				return `<p>Practice Round</p><p>Bet on Red That is Switched</p><p>&nbsp;</p>` + format_image_button("lights/Slide1") +
				`<div>` + format_image_button("observe/Slide1") + `</div>`;
			},

			button_html: function () {
				return format_image_button("lights/Slide2");
			},
			//margin_vertical: '18px',

			prompt: `<p>Click on one of the pictures to continue</p>`,

		};
		//timeline.push(iti)
		instructions_timeline.push(teach_bet_unsuccessful_red)
		
		var feedback_bet_unsuccessful_red = {
			type: jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus: `<p>Great!</p><p>Here is an explanation of feedback you might get after placing a bet that is switched:</p>`,
			button_html: `<img style="max-width:675px;" src="stim/instructions/Slide7.JPG">`,
			prompt: `<p>Click on the pictures to continue</p>`,
		};

		instructions_timeline.push(feedback_bet_unsuccessful_red)

		quiz_prompts = [ "In order to place a bet, you clicked on one of which pair of buttons in the two preceding rounds?", 
		"What does the appearance of a ring with a circle mean after you placed a bet? <img style='max-width: 75px;' src='stim/lights/Slide3.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide4.JPG'></src>",
		"What does the appearance of a circle with a coin and question mark (but without a ring) after you place a bet signify? <img style='max-width: 75px;' src='stim/lights/Slide7.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide8.JPG'></src>"
			]

		quiz_options = [
						[
							"<img style='max-width: 75px;' src='stim/lights/Slide1.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide2.JPG'></src>",
							"<img style='max-width: 75px;' src='stim/observe/Slide1.JPG'></src><img style='max-width: 75px;' src='stim/observe/Slide2.JPG'></src>",
						],
						[
							"The corresponding light lit up, meaning that you earned a coin",
							"The corresponding light lit up, meaning that you would have earned a coin if you had bet on it in that round, but actually didn't win any coins this round",
							"You successfully placed a bet on that light, implying potential winnings if the light paid out this round",
							"You tried to bet on that color but didn't succeed, so your bet was switched, implying potential winnings if the other light paid out this round", 
						] ,
						[
							"The corresponding light lit up, meaning that you earned a coin",
							"You did not bet on this light, meaning that you won't earn a coin if it lit up",
							"You tried to bet on the other color but didn't succeed so that your bet was switched to this one, meaning you gained reward if this light lit up that round (which is unknown at the moment)",
							"You tried to bet on that color but didn't succeed, i.e. your bet was switched, meaning you gained reward if the other light lit up that round (which is unknown at the moment)", 
						]
		]

		quiz_idxs_correct = [0, 3, 2]

		solutions_pictures = ["",
			"<img style='max-width: 75px;' src='stim/lights/Slide3.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide4.JPG'></src>",
			"<img style='max-width: 75px;' src='stim/lights/Slide7.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide8.JPG'></src>"
		]

		instructions_timeline = instructions_timeline.concat(create_quiz(quiz_prompts, quiz_options, quiz_idxs_correct, solutions_pictures))

		var instructions_uncertainty = {
			type : jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus: `<p>Great! Now, let's review the sources of uncertainty on the game before we begin.` + 
				`<ol  style='text-align:left'>` +
					`<li>The 'lucky' light can switch randomly between rounds (with roughly 10\% chance, so it switches on average once every 10 rounds). Both colors are equally likely to show up overall, so <b><i>if you want to track the lucky light, you will need to observe periodically from time to time</b></i>.</li>` +
					`<li>Your ability to place a bet successfully may vary, but it never switches more than 50% of the time so that it is never worth it to place the bet on the other light in the hopes that it is switched.</li>` +
					`<li><b>By observing periodically and tracking the lucky light in sets where you can control your bets very well, you can earn very high scores. <i>However, observing in sets where you don't have control will reduce the total amount of reward you can earn due to missed opportunities (since you can't place your bets as successfully, it's not valuable to know which light is the lucky one).</i></b> Generally, a good strategy is to observe more on sets where you can place your bets successfully easily, since you will be able to capitalize on this information, and to observe less in rounds where you don't (i.e. it is ok to bet randomly in those cases, since you won't be able to place your bets on the arms you want in any case).</li>` +
				`</ol>`
		};
		instructions_timeline.push(instructions_uncertainty);

		quiz_uncertainty_prompts = [ //"How stable is the 'lucky' light?",
		"What is a good strategy to track the lucky light in sets where you have a lot of control?",
		"How likely are you to place a bet on the light that you want?", 
		"When you are in a set where you have no control over on which light you are placing your bet, should you spend time observing?",
        "Will you ever be able to predict when your bet will be switched to the other light, i.e. does it ever make sense to try to place a light on the wrong one in the hopes that your bet will fail and be switched to the right one instead?",
			]

		quiz_uncertainty_options = [
						[
							"Observing a lot at the beginning and then not anymore",
                            "Observing every once in a while, as observing will tell me the lucky light for a certain period, but the lucky light can switch after that",
                            "It is impossible to know what the lucky light is so I can just guess randomly",
						],
						[
							"You always successfully place a bet on the light you want",
							"With a chance constant during sets and always less or equal to 50%, you may randomly accidentally place your bet on the other light. It's never beneficial to intentionally bet on the wrong light, and rather I should adjust my observation patterns",
                            "With a chance constant during sets and frequently >50%, you may randomly accidentally place your bet on the other light. This switching is predictable, so it's often beneficial to bet on the wrong light",
                            "You rarely place a successful bet; when unsuccessful, no bet is placed, and you earn no money in that round",
						],
						[
							"No, since you won't be able to place your bets successfully on the light you want in low-control sets, it's not worth knowing which light is the lucky one so it is best to just place bets randomly and not waste any time observing",
							"Yes, you should observe more because you can't bet as effectively",
                        ],
                        [
                            "Yes",
                            "No"
                        ],
		]

		quiz_uncertainty_idxs_correct = [1, 1, 0, 1, ]

		solutions_uncertainty_pictures = ["",
			"<img style='max-width: 75px;' src='stim/lights/Slide3.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide4.JPG'></src>",
			"<img style='max-width: 75px;' src='stim/lights/Slide7.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide8.JPG'></src>"
		]

		instructions_timeline = instructions_timeline.concat(create_quiz(quiz_uncertainty_prompts, quiz_uncertainty_options, quiz_uncertainty_idxs_correct, solutions_uncertainty_pictures))

		var instructions_structure = {
			type : jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus: `<p>Great! On the next page, there will be a final quiz on the game's structure. ` + 
				`First, let's review some key facts:</p> ` +
				`<ul style='text-align:left'>` +
				`<li>The money you amassed during a set of 50 rounds is revealed at the end of the set</li>` +
				`<li>A detailed overview is provided after the first ` + params.n_episodes_train + ` sets</li>` +
				`<li>In total, you will play ` + (params.n_episodes_train + params.n_episodes_test).toString() + `  sets of 50 rounds each</li>` +
				`</ul>`
		};
		instructions_timeline.push(instructions_structure);

		quiz_structure_prompts = [ 
		"In sets where you have less control, will the lucky light switch more often, less often, or at the same rate?",
		"When can you expect to see how much money you will have won for your bets?",
		"When will you receive feedback on the sequence of lights that lit up, as well as your choices in the corresponding trials?"
			]

		quiz_structure_options = [
						[
                            "More often",
                            "At the same rate, as the speed at which the lucky light switches is the same throughout the game, whereas my ability to place my bets where I intend changes between sets",
							"Less often"
                        ],
						[
							"Immediately after placing the bet",
							"At the end of a set of bets, i.e. after 50 individual rounds of choosing to either observe or bet",
							"At the end of the experiment",
							"Never", 
						] ,
						[
							"At the end of every round",
							//"At the end of a set (of 50 rounds) for the first 4 rounds only",
							"At the end of a set (of 50 rounds)",
							"At the end of the experiment",
							"Never", 
						]
		]

		quiz_structure_idxs_correct = [1, 1, 1, 1]

		solutions_structure_pictures = ["",
			"<img style='max-width: 75px;' src='stim/lights/Slide3.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide4.JPG'></src>",
			"<img style='max-width: 75px;' src='stim/lights/Slide7.JPG'></src><img style='max-width: 75px;' src='stim/lights/Slide8.JPG'></src>"
		]

		instructions_timeline = instructions_timeline.concat(create_quiz(quiz_structure_prompts, quiz_structure_options, quiz_structure_idxs_correct, solutions_structure_pictures))

		var end_instructions = {
			type : jsPsychHtmlButtonResponse,
			choices: ["Continue"],
			stimulus : `<p>"Fantastic," the employee says. "It looks like you have understood the game. So, let's go ahead and get started!"</p>`,
			on_start : set_html_style_instructions,
			on_finish : set_html_style_normal
		}
		instructions_timeline.push(end_instructions)
	}

    var bloc = {
        timeline : instructions_timeline,
    }

    return bloc
}