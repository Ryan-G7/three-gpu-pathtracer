import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rand2, RNG_INDEX_SCATTER_DIRECTION } from '../nodes/random.wgsl.js';
import { diffuseDirectionFunc } from '../nodes/sampling.wgsl.js';

/**
 * Defines a material sampled by the pathtracer
 */
export class PathtracingMaterial {

	/**
	 *
	 * Called once per material
	 * Adds ability to initialize state
	 *
	 */
	init( /* renderer */ ) {

	}

	/**
	 *
	 * Must return a bsdf sampling function node with signature
	 * ( worldView: vec3f, surface: Surface ) -> ScatterRecord
	 *
	 */
	getBsdfNode() {

		return wgslTagFn`

			fn bsdfSample( worldWo: vec3f, surf: SurfaceRecord ) -> ScatterRecord {

				var record: ScatterRecord;

				let wo = normalize( surf.normalInvBasis * worldWo );
				let wi = ${ diffuseDirectionFunc }( wo, ${ rand2 }( ${ RNG_INDEX_SCATTER_DIRECTION } ) );
				record.color = surf.color * max( wi.z, 0.0 ) / PI;
				record.pdf = max( wi.z, 0.0 ) / PI;
				record.specularPdf = 0.0;
				record.direction = normalize( surf.normalBasis * wi );

				return record;

			}

		`;

	}

	/**
	 *
	 * Must return a BSDF evaluation function node with signature
	 * ( worldView: vec3f, worldLight: vec3f, surface: Surface ) -> ScatterRecord
	 *
	 */
	getBsdfEvalNode() {

		return wgslTagFn`

			fn bsdfEval( worldWo: vec3f, worldWi: vec3f, surf: SurfaceRecord ) -> ScatterRecord {

				var record: ScatterRecord;
				let wi = normalize( surf.normalInvBasis * worldWi );
				let cosTheta = max( wi.z, 0.0 );
				record.color = surf.color * cosTheta / PI;
				record.direction = worldWi;
				record.pdf = cosTheta / PI;
				record.specularPdf = 0.0;
				return record;

			}

		`;

	}

	getBsdfNodes() {

		return { sample: this.getBsdfNode(), evaluate: this.getBsdfEvalNode() };

	}

	getData() {

		const { sample, evaluate } = this.getBsdfNodes();
		return {

			bsdfSample: sample,
			bsdfEval: evaluate,

		};

	}

}
