import { IndirectStorageBufferAttribute, Matrix3, StorageTexture, DataTexture } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { uniform, storage, textureStore, globalId, texture, sampler } from 'three/tsl';
import { queuedRayStruct, queuedHitStruct } from './structs.js';
import { proxy, proxyFn, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { misHeuristicFn, sampleEnvironmentDirectionFn, weightedAlphaBlendFn } from '../../nodes/sampling.wgsl.js';
import { isTerminatingScatterFunc, offsetRayOriginFunc } from '../../nodes/utils.wgsl.js';
import { rand2, rngInit, RNG_INDEX_ENVIRONMENT_LIGHT } from '../../nodes/random.wgsl.js';

export class ProcessHitsKernel extends ComputeKernel {

	constructor( ) {

		const params = {
			bvhData: { value: null },
			material: { value: null },

			prevOutputTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadOnly(),
			outputTarget: textureStore( new StorageTexture( 1, 1 ) ).toWriteOnly(),
			sampleCountTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadWrite(),

			// settings
			smoothNormals: uniform( 1 ),
			bounces: uniform( 1 ),

			// rays
			rayQueue: storage( new IndirectStorageBufferAttribute( 1, queuedRayStruct.getLength() ), queuedRayStruct ),
			hitQueue: storage( new IndirectStorageBufferAttribute( 1, queuedHitStruct.getLength() ), queuedHitStruct ),
			queueSizes: storage( new IndirectStorageBufferAttribute( 4, 1 ), 'u32' ).toAtomic(),

			textures: texture( new DataTexture() ),
			textureSampler: sampler( new DataTexture() ),

			// environment
			envMap: texture( new DataTexture() ),
			envMapSampler: sampler( new DataTexture() ),
			envMarginalWeights: texture( new DataTexture() ),
			envConditionalWeights: texture( new DataTexture() ),
			envTotalSum: uniform( 0 ),
			envMapRotation: uniform( new Matrix3() ),
			envMapIntensity: uniform( 1 ),

			globalId: globalId,
		};

		const raycastOutput = proxy( 'bvhData.value.fns.raycastFirstHit.outputType', params );
		const raycastFirstHitFn = proxyFn( 'bvhData.value.fns.raycastFirstHit', params );
		const sampleTrianglePointFn = proxyFn( 'bvhData.value.fns.sampleTrianglePoint', params );
		const getSurfaceRecordFn = proxyFn( 'bvhData.value.fns.getSurfaceRecord', params );
		const bsdfSampleFn = proxyFn( 'material.value.bsdfSample', params );
		const bsdfEvalFn = proxyFn( 'material.value.bsdfEval', params );

		const fn = wgslTagFn/* wgsl */`

			fn compute(
				// settings
				smoothNormals: u32,
				bounces: u32,

				// environment
				envMap: texture_2d<f32>,
				envMapSampler: sampler,
				envMarginalWeights: texture_2d<f32>,
				envConditionalWeights: texture_2d<f32>,
				envTotalSum: f32,
				envMapRotation: mat3x3f,
				envMapIntensity: f32,

				globalId: vec3u
			) -> void {

				let rayQueue = &${ params.rayQueue };
				let hitQueue = &${ params.hitQueue };
				let queueSizes = &${ params.queueSizes };

				let materials = &${ proxy( 'bvhData.value.storage.materials', params ) };
				let transforms = &${ proxy( 'bvhData.value.storage.transforms', params ) };
				let envInfo = EnvironmentInfo( envMapRotation, envMapIntensity, 0.0 );

				// skip any rays invocations beyond the ray count
				let hitQueueCapacity = arrayLength( hitQueue );
				let hitIndex = ( globalId.x + atomicLoad( &queueSizes[ 2 ] ) );
				if ( hitIndex >= atomicLoad( &queueSizes[ 3 ] ) ) {

					return;

				}

				// get the ray info
				let ACTIVE_FLAG = 0xF0000000u;
				let input = hitQueue[ hitIndex ];
				let indexUV = vec2u( input.pixel_x, input.pixel_y );
				${ rngInit }( indexUV.xy, input.seed, input.currentBounce );

				let object = transforms[ input.objectIndex ];
				var material = materials[ object.materialIndex ];

				// apply per-object colors
				material.color *= object.color.rgb;
				material.opacity *= object.color.a;

				let barycoord = vec3( input.barycoord, 1.0 - input.barycoord.x - input.barycoord.y );
				var vertexData = ${ sampleTrianglePointFn }( barycoord, input.indices.xyz );
				vertexData.normal = normalize( transpose( object.inverseMatrixWorld ) * vertexData.normal );
				vertexData.position = object.matrixWorld * vertexData.position;

				let surface = ${ getSurfaceRecordFn }( material, vertexData, input.side, input.normal );

				let scatterRec = ${ bsdfSampleFn }( input.view, surface );

				var resultColor = input.resultColor + vec4f( input.throughputColor * surface.emission, 0.0 );

				let envSample = ${ sampleEnvironmentDirectionFn }(
					envMap, envMapSampler, envMarginalWeights, envConditionalWeights,
					envInfo, envTotalSum, ${ rand2 }( ${ RNG_INDEX_ENVIRONMENT_LIGHT } )
				);
				let envBsdf = ${ bsdfEvalFn }( input.view, envSample.direction, surface );
				let orientedNormal = input.normal * input.side;
				if ( envSample.pdf > 0.0 && envBsdf.pdf > 0.0 && dot( envSample.direction, orientedNormal ) > 0.0 ) {

					let shadowOrigin = ${ offsetRayOriginFunc }( vertexData.position.xyz, envSample.direction, input.normal );
					let shadowRay = Ray( shadowOrigin, envSample.direction );
					var shadowHit: ${ raycastOutput };
					if ( ! ${ raycastFirstHitFn }( shadowRay, &shadowHit ) ) {

						let misWeight = ${ misHeuristicFn }( envSample.pdf, envBsdf.pdf );
						resultColor += vec4f( input.throughputColor * envSample.color * envBsdf.color * misWeight / envSample.pdf, 0.0 );

					}

				}

				let isTerminated = input.currentBounce >= bounces || ${ isTerminatingScatterFunc }( scatterRec );

				if ( isTerminated ) {

					// terminate ray, write color
					let sampleCount = ( textureLoad( ${ params.sampleCountTarget }, indexUV ).r & ( ~ ACTIVE_FLAG ) ) + 1;
					let prevColor = textureLoad( ${ params.prevOutputTarget }, indexUV );
					let blendedColor = ${ weightedAlphaBlendFn }( prevColor, resultColor, 1.0 / f32( sampleCount ) );
					textureStore( ${ params.sampleCountTarget }, indexUV, vec4( sampleCount ) );
					textureStore( ${ params.outputTarget }, indexUV, blendedColor );

				} else {

					let rayQueueCapacity = arrayLength( rayQueue );
					let index = atomicAdd( &queueSizes[ 1 ], 1 ) % rayQueueCapacity;
					rayQueue[ index ].origin = ${ offsetRayOriginFunc }( vertexData.position.xyz, scatterRec.direction, input.normal );
					rayQueue[ index ].direction = scatterRec.direction;
					rayQueue[ index ].pdf = scatterRec.pdf;
					rayQueue[ index ].pixel = indexUV;
					rayQueue[ index ].throughputColor = input.throughputColor * scatterRec.color / scatterRec.pdf;
					rayQueue[ index ].currentBounce = input.currentBounce + 1;
					rayQueue[ index ].resultColor = resultColor;
					rayQueue[ index ].seed = input.seed;

				}

			}`;

		super( fn( params ) );

		this.defineUniformAccessors( params );

	}

}
