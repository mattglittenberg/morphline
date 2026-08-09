/*
 * Generate synthetic FreeSurfer stats fixtures.
 *
 * Runs only when --input is absent, which is what makes `-profile test` work
 * on a cold clone with no external data present (BUILD_PLAN §2.7).
 */
process GENERATE_FIXTURES {
    tag 'fixtures'
    publishDir "${params.outdir}/fixtures", mode: 'copy', enabled: false

    input:
    path config

    output:
    path 'fixtures'    , emit: fixtures
    path 'versions.yml', emit: versions

    script:
    """
    morphline fixtures generate --config ${config} --outdir fixtures

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        morphline: \$(morphline version)
    END_VERSIONS
    """
}
