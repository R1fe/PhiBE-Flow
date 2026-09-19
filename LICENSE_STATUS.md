# License Status: Approval Required Before Publication

No project-wide license has been selected by the rights holder yet. This file
does not grant a license to original project contributions.

## Newly identified GPE restriction

The supplied GPE source's LICENSE and README identify the Regents of the
University of Minnesota as the copyright holder, restrict use to specified
educational/research institutions, and require prior approval to redistribute.
Upstream: https://github.com/wonjunee/GPE_codes

This is not an MIT or GPL grant. A GPL-3.0 label on this project cannot override
those restrictions. Previously copied GPE encoder/decoder definitions have
been removed from the current source tree. Readers must obtain GPE separately
under its own terms; see EXTERNAL_CODECS.md. Old Git commits may still contain
removed source and are not a cleaned distribution history.
An anonymous proxy also distributes code; keeping the origin private does not
resolve licensing once the proxy exposes its contents. Do not copy additional
GPE source or weights into a release while this is unresolved.

Possible routes are written permission covering the intended redistribution,
or an external-dependency design reviewed for its own licensing implications.
Merely moving an import to another file is not a compatibility determination.

The included RIVER-derived implementation is distributed upstream under GPL-3.0;
its license text is retained in `licenses/RIVER-GPL-3.0.txt`. Do not label this
combined artifact MIT-only. Obtain the rights holder's confirmation of a
compatible project-wide license before publication.

Taming Transformers and the diffusion U-Net carry MIT notices in `licenses/`.
Those notices identify third-party authors, not the submitting authors, and
must not be deleted to achieve anonymity.

Before release, confirm provenance/redistribution rights for the Acrobot/GPE
and NSE source contributions, the optional FVD helper and any published data
or model checkpoints. See THIRD_PARTY_NOTICES.md. Consult the relevant rights
holders if the applicable license or permission is unclear.

This is a release-preparation checklist, not legal advice.
