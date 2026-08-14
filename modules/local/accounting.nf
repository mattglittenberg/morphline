/*
 * Data accounting funnel (§1.6).
 *
 * Exits non-zero if the funnel does not reconcile, so unexplained data loss
 * fails the pipeline rather than printing a warning nobody reads.
 */
process ACCOUNTING {
    tag 'accounting'
    publishDir "${params.outdir}", mode: 'copy', pattern: '*.{parquet,json}'

    // ingest_counters.json is staged but never named in the command: `morphline
    // account` looks for it beside --observations. Without it the stage falls
    // back to re-deriving discovery counts, which is exactly the "never derived"
    // rule this funnel exists to uphold — and it fails silently, because on
    // clean inputs the derived numbers agree.
    input:
    path observations
    path ingest_counters
    path failures
    path qc_observations
    path model_results
    path config
    path dataset

    output:
    path 'accounting_funnel.parquet', emit: funnel
    path 'accounting.json'          , emit: json
    path 'versions.yml'             , emit: versions

    script:
    def failure_args = failures.collect { "--failures ${it}" }.join(' ')
    """
    morphline account \\
        --config ${config} \\
        --indir ${dataset} \\
        --observations ${observations} \\
        --qc-observations ${qc_observations} \\
        --model-results ${model_results} \\
        ${failure_args} \\
        --outdir .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        morphline: \$(morphline version)
    END_VERSIONS
    """
}
