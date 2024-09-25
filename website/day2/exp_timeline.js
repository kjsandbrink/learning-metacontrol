// Define main experimental timeline
// This adds all the trials to a timeline array
function returnTimeline(params) {
    let timeline = []
    timeline.push(blocStart(params))
    // One bloc
    timeline.push(blocMain(params))
    timeline.push(blocEnd(params))
    return timeline
}
